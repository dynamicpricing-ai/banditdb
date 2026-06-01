use ndarray::Array1;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::Ordering;
use crate::state::{ArmState, NeuralLinUCBConfig};
use candle_core::{DType, Device, Tensor};
use candle_nn::{AdamW, Linear, Module, Optimizer, ParamsAdamW, VarBuilder, VarMap};

// ---------------------------------------------------------------------------
// Device selection
// ---------------------------------------------------------------------------

/// Picks a compute device based on the BANDITDB_DEVICE env var.
/// Priority when unset or "auto": CUDA → Metal → CPU.
/// Examples: BANDITDB_DEVICE=cpu  BANDITDB_DEVICE=cuda  BANDITDB_DEVICE=cuda:1
pub fn select_device() -> Device {
    let hint = std::env::var("BANDITDB_DEVICE").unwrap_or_default();
    match hint.as_str() {
        "cpu"   => Device::Cpu,
        "metal" => try_metal(),
        "cuda"  => try_cuda(0),
        s if s.starts_with("cuda:") => s[5..].parse().map(try_cuda).unwrap_or_else(|_| try_cuda(0)),
        _ => {
            // Auto-detect: CUDA → Metal → CPU
            #[cfg(feature = "cuda")]
            if let Ok(d) = Device::new_cuda(0) { return d; }
            #[cfg(feature = "metal")]
            if let Ok(d) = Device::new_metal(0) { return d; }
            Device::Cpu
        }
    }
}

#[cfg(feature = "cuda")]
fn try_cuda(ordinal: usize) -> Device {
    Device::new_cuda(ordinal).unwrap_or_else(|_| {
        tracing::warn!(ordinal, "neural: CUDA unavailable — falling back to CPU");
        Device::Cpu
    })
}
#[cfg(not(feature = "cuda"))]
fn try_cuda(_ordinal: usize) -> Device {
    tracing::warn!("neural: CUDA requested but binary not compiled with --features cuda");
    Device::Cpu
}

#[cfg(feature = "metal")]
fn try_metal() -> Device {
    Device::new_metal(0).unwrap_or_else(|_| {
        tracing::warn!("neural: Metal unavailable — falling back to CPU");
        Device::Cpu
    })
}
#[cfg(not(feature = "metal"))]
fn try_metal() -> Device {
    tracing::warn!("neural: Metal requested but binary not compiled with --features metal");
    Device::Cpu
}

const BUFFER_CAP: usize = 200_000;

pub struct NeuralLinUCBState {
    varmap:        VarMap,
    layers:        Vec<Linear>,
    w0:            Vec<Tensor>,   // frozen copy of initial weights for Algorithm 2 regularization
    device:        Device,
    pub context_dim:   usize,
    pub embed_dim:     usize,
    retrain_every: usize,
    retrain_steps: usize,
    learning_rate: f64,
    lambda:        f64,
    pub reward_count:  usize,
    pub buffer:    VecDeque<(Vec<f64>, String, f64, f64)>, // context, arm_id, reward, propensity
    pub last_retrain_losses: Vec<f32>,
}

impl NeuralLinUCBState {
    pub fn new(cfg: &NeuralLinUCBConfig) -> candle_core::Result<Self> {
        let device = select_device();
        tracing::info!(device = ?device, "neural: device selected");
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let mut layers: Vec<Linear> = Vec::new();
        let mut in_dim = cfg.context_dim;

        for i in 0..cfg.hidden_layers {
            layers.push(candle_nn::linear(in_dim, cfg.hidden_dim, vb.pp(format!("h{i}")))?);
            in_dim = cfg.hidden_dim;
        }
        layers.push(candle_nn::linear(in_dim, cfg.embed_dim, vb.pp("embed"))?);

        // Save W₀: independent copies of initial weights (frozen throughout training)
        let w0 = varmap.all_vars().iter()
            .map(|v| copy_tensor(v.as_tensor(), &device))
            .collect::<candle_core::Result<Vec<_>>>()?;

        Ok(Self {
            varmap,
            layers,
            w0,
            device,
            context_dim:   cfg.context_dim,
            embed_dim:     cfg.embed_dim,
            retrain_every: cfg.retrain_every,
            retrain_steps: cfg.retrain_steps,
            learning_rate: cfg.learning_rate,
            lambda:        cfg.lambda,
            reward_count:        0,
            buffer:              VecDeque::with_capacity(BUFFER_CAP),
            last_retrain_losses: Vec::new(),
        })
    }

    /// L2-normalised last-layer embedding — the h(x; W) used by Algorithm 1.
    pub fn embed(&self, context: &Array1<f64>) -> Array1<f64> {
        self.try_embed(context).unwrap_or_else(|_| Array1::zeros(self.embed_dim))
    }

    fn try_embed(&self, context: &Array1<f64>) -> candle_core::Result<Array1<f64>> {
        let x: Vec<f32> = context.iter().map(|&v| v as f32).collect();
        let t = Tensor::from_slice(&x, (1, x.len()), &self.device)?;
        let h = self.forward_normalized(&t)?.squeeze(0)?;
        let vals: Vec<f32> = h.to_vec1()?;
        Ok(Array1::from_vec(vals.iter().map(|&v| v as f64).collect()))
    }

    /// Forward pass with L2 normalisation (used for both scoring and training).
    fn forward_normalized(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let mut h = x.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h)?;
            if i < self.layers.len() - 1 {
                h = h.relu()?;
            }
        }
        // h / (||h||₂ + ε)  — ε via affine to avoid division by zero
        let norm = h.sqr()?.sum_keepdim(1)?.affine(1.0, 1e-8)?.sqrt()?;
        h.broadcast_div(&norm)
    }

    pub fn push(&mut self, context: Vec<f64>, arm_id: String, reward: f64, propensity: f64) {
        if self.buffer.len() >= BUFFER_CAP {
            self.buffer.pop_front();
        }
        self.buffer.push_back((context, arm_id, reward, propensity));
        self.reward_count += 1;
    }

    pub fn should_retrain(&self) -> bool {
        self.reward_count >= self.retrain_every && !self.buffer.is_empty()
    }

    /// Algorithm 2: J gradient descent steps minimising
    ///   L(W) = ½ · mean(θᵀh(x;W) − r)²  +  (m·λ/2)‖W − W₀‖²_F
    ///
    /// All buffer interactions are stacked into batch tensors before the loop so each
    /// step is a single GPU kernel launch rather than N sequential host→device transfers.
    pub fn retrain(&mut self, arm_states: &HashMap<String, ArmState>) -> candle_core::Result<()> {
        if self.buffer.is_empty() { return Ok(()); }

        let buffer: Vec<_> = self.buffer.iter().cloned().collect();
        let n = buffer.len();
        let n_params: f64 = self.varmap.all_vars().iter()
            .map(|v| v.as_tensor().elem_count())
            .sum::<usize>() as f64;

        // Build batch tensors once — one host→device transfer before the training loop.
        //   batch_x:     (n, context_dim)
        //   batch_theta: (n, embed_dim)   — θ of the selected arm per interaction
        //   batch_r:     (n, 1)           — observed rewards
        let flat_x: Vec<f32> = buffer.iter()
            .flat_map(|(ctx, _, _, _)| ctx.iter().map(|&v| v as f32))
            .collect();
        let batch_x = Tensor::from_slice(&flat_x, (n, self.context_dim), &self.device)?;

        let flat_theta: Vec<f32> = buffer.iter()
            .flat_map(|(_, arm_id, _, _)| {
                arm_states.get(arm_id)
                    .map(|a| a.theta.iter().map(|&v| v as f32).collect::<Vec<_>>())
                    .unwrap_or_else(|| vec![0.0f32; self.embed_dim])
            })
            .collect();
        let batch_theta = Tensor::from_slice(&flat_theta, (n, self.embed_dim), &self.device)?;

        let rewards: Vec<f32> = buffer.iter().map(|(_, _, r, _)| *r as f32).collect();
        let batch_r = Tensor::from_slice(&rewards, (n, 1), &self.device)?;

        let mut opt = AdamW::new(
            self.varmap.all_vars(),
            ParamsAdamW { lr: self.learning_rate, ..Default::default() },
        )?;

        let mut losses: Vec<f32> = Vec::with_capacity(self.retrain_steps);

        for _ in 0..self.retrain_steps {
            // One forward pass on the whole batch → (n, embed_dim)
            let batch_h = self.forward_normalized(&batch_x)?;

            // Per-sample predicted reward: row-wise θᵀh → (n, 1)
            let preds = batch_h.mul(&batch_theta)?.sum_keepdim(1)?;

            // ½ · MSE  +  (m·λ/2)·‖W − W₀‖²
            let loss = preds.sub(&batch_r)?.sqr()?.mean_all()?
                .affine(0.5, 0.0)?
                .add(&self.w0_reg(n_params)?)?;

            losses.push(loss.to_scalar::<f32>()?);
            opt.backward_step(&loss)?;
        }

        self.last_retrain_losses = losses;
        self.reward_count = 0;
        Ok(())
    }

    fn w0_reg(&self, _n_params: f64) -> candle_core::Result<Tensor> {
        let scale = self.lambda / 2.0;
        let all_vars = self.varmap.all_vars();
        let mut reg = Tensor::zeros((), DType::F32, &self.device)?;

        for (var, w0) in all_vars.iter().zip(self.w0.iter()) {
            let diff = var.as_tensor().sub(w0)?;
            reg = reg.add(&diff.sqr()?.sum_all()?)?;
        }
        reg.affine(scale, 0.0)
    }

    /// Re-accumulate per-arm LinUCB statistics in the new embedding space using
    /// Sherman-Morrison (same incremental updates as the online path — no matrix inversion).
    pub fn reaccumulate(&self, old_arms: &HashMap<String, ArmState>) -> HashMap<String, ArmState> {
        let mut new_arms: HashMap<String, ArmState> = old_arms.keys()
            .map(|k| (k.clone(), ArmState::new(self.embed_dim)))
            .collect();

        // Preserve counters from old states (they are stats, not algorithm state)
        for (arm_id, new_state) in new_arms.iter_mut() {
            if let Some(old) = old_arms.get(arm_id) {
                new_state.prediction_count.store(old.prediction_count.load(Ordering::Relaxed), Ordering::Relaxed);
                new_state.reward_count.store(old.reward_count.load(Ordering::Relaxed), Ordering::Relaxed);
                new_state.total_reward.store(old.total_reward.load(Ordering::Relaxed), Ordering::Relaxed);
            }
        }

        for (ctx, arm_id, reward, _) in &self.buffer {
            if let Some(state) = new_arms.get_mut(arm_id) {
                let h = self.embed(&Array1::from_vec(ctx.clone()));
                state.update(&h, *reward);
            }
        }

        new_arms
    }

    pub fn save(&self, path: &str) -> candle_core::Result<()> {
        self.varmap.save(path)
    }

    pub fn load(&mut self, path: &str) -> candle_core::Result<()> {
        self.varmap.load(path)
    }
}

/// Creates an independent tensor copy with the same shape and data.
fn copy_tensor(t: &Tensor, device: &Device) -> candle_core::Result<Tensor> {
    let shape = t.shape().clone();
    let flat: Vec<f32> = t.flatten_all()?.to_vec1()?;
    Tensor::from_vec(flat, shape, device)
}
