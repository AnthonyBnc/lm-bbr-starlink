# So sánh công thức: paper SLM-BBR / paper của bạn ↔ code trong repo `lm-bbr-starlink`

*Tài liệu này đối chiếu từng công thức toán học xuất hiện trong (A) paper nguồn **SLM-BBR** (`project_sources/2607.07142v1 copy.pdf`, arXiv:2607.07142v1, De Silva, Pokhrel, Kua — chính là ref. [1] mà `docs/PAPER_DRAFT.md` trích dẫn) và (B) paper của bạn (`docs/PAPER_DRAFT.md`, "Quantum-Enhanced Language-Model-Based Control for BBR over LEO Satellite Internet") với phần code thực sự triển khai chúng trong repo. Không có công thức nào bị bỏ sót — tổng cộng 27 công thức đánh số của paper SLM-BBR (Eq. 1–27) và các công thức của phần Quantum Head / loss trong paper của bạn.*

---

## 0. Hai "paper" đang được nhắc tới

| | Paper | File | Vai trò |
|---|---|---|---|
| **Paper nguồn (A)** | *Small Language Model-based Control for BBR over Low Earth Orbit Satellite Internet* | `project_sources/2607.07142v1 copy.pdf` | Định nghĩa toàn bộ pipeline: phase detection, action space, reward, surrogate throughput/retransmission model, LoRA, loss. Repo **reproduce** paper này (xem `ADR-0001`, `ADR-0012`). |
| **Paper của bạn (B)** | *Quantum-Enhanced Language-Model-Based Control for BBR over LEO Satellite Internet* | `docs/PAPER_DRAFT.md` | Kế thừa nguyên vẹn pipeline của (A), chỉ thay **task head** (classical → quantum) và **đặc tả lại** loss dưới dạng phase-masked cross-entropy tường minh. |

Repo có `utils/starlink_preprocessing.py`, `utils/bbr.py`, `utils/tokyo_evaluation.py` triển khai công thức của (A); và `plm_special/quantum_head.py`, `plm_special/classical_head.py`, `utils/training_metrics.py` triển khai công thức của (B).

---

## PHẦN A — Paper SLM-BBR gốc (Eq. 1–27) ↔ code

### A1. Phase detection (Algorithm 1, Eq. 1)

**Paper:**
```
d_i = b_i - b̄_i                                  (Eq. 1)
σ_d = std(d_i)
d↑ = κ·σ_d ,  d↓ = -κ·σ_d      (κ = 0.7, Table I)
```
`ProbeBW_UP` = local maximum với `d_i > d↑`; `ProbeBW_DOWN` = local minimum kế tiếp với `d_i < d↓`; 6 sample sau `DOWN` bị ép về `CRUISE`.

**Code:** `utils/starlink_preprocessing.py :: detect_bbr_phases()`
```python
deviation = throughput - _rolling_mean(throughput, half_window=10)   # Eq. 1
sigma = std(deviation)
upper = 0.7 * sigma; lower = -upper                                  # κ = 0.7
... local_maxima / local_minima (so sánh 2 hàng xóm) ...
cruise_end = down_index + 1 + 6                                      # 6 mẫu CRUISE
```
**Kết quả:** ✅ khớp đúng công thức và hằng số (κ=0.7, 6 mẫu). Paper **không** định nghĩa chính xác thế nào là "local maximum/minimum" (bao nhiêu điểm lân cận) — code tự chọn quy tắc so sánh với 1 hàng xóm mỗi bên, và điều này được ghi lại rõ ràng như một giả định trong `ADR-0001` ("Local extrema use adjacent samples"). Cửa sổ rolling-mean nửa-window `w=10` (→ 21 mẫu) cũng là một giả định được `ADR-0001` phê duyệt, vì paper chỉ nói "w=10" cho rolling percentile chứ không nói rõ áp dụng cho rolling mean của phase detector.

### A2. Continuous gain estimate & discretisation (Eq. 2–3)

**Paper:**
```
G↑_i = 3 / (B̄_i + 2)          G↓_i = (B̄_i + 1) / 2       (Eq. 2)
B̄_i = b_i / max_j(b_j)

G_i = argmin_{g∈S_UP}  |g - G↑_i|     nếu phase = UP
      argmin_{g∈S_DOWN}|g - G↓_i|     nếu phase = DOWN     (Eq. 3)
      1.00                            nếu phase = CRUISE
```

**Code:** `_equation_2_3_action()`
```python
if phase == BW_UP:     target_gain = 3.0 / (utilization_proxy + 2.0)
elif phase == BW_DOWN: target_gain = (utilization_proxy + 1.0) / 2.0
elif phase == BW_CRUISE: target_gain = 1.0
action_index = _nearest_action_index(target_gain, phase)   # nearest, tie → index nhỏ hơn
```
**Kết quả:** ✅ khớp 1:1 kể cả hằng số (3, 2, 1, 2). `utilization_proxy = b_i / max_j(b_j)` cũng khớp `B̄_i`. Tie-break "chọn index nhỏ hơn" là chi tiết paper không nói rõ, do code tự bổ sung.

### A3. Reward components — throughput/retransmission utilisation (Eq. 4–13)

| Paper | Công thức | Code | Trạng thái |
|---|---|---|---|
| Eq. 4 | `B_i = min(b_i / B_ref_i, 1)` | `rate_utilization = min(throughput/throughput_ref, 1)` | ✅ khớp (đổi tên biến) |
| Eq. 5 | `τ̄_i = tanh(τ_i / τ̄_ref_i)` | **không có `tanh` nào trong `starlink_preprocessing.py`/`tokyo_evaluation.py`** | ❌ **không triển khai** — bị thay thế hoàn toàn bởi surrogate Eq. 27 khi tính reward (xem A5) |
| Eq. 6 | `B_ref_i = Q_0.95(b_{i-w:i+w})` | `throughput_ref = _rolling_percentile(throughput, w=10)` | ✅ khớp |
| Eq. 7 | `τ̄_ref_i = Q_0.95(τ_{i-w:i+w}) + 1` | `retransmit_ref = _rolling_percentile(retransmits, w=10) + 1.0` | ✅ khớp |
| Eq. 8 | `q_ref_i = min(Q_0.95(qd_{i-w:i+w}), 0.15·RTT_min)` | `queue_ref = min(_rolling_percentile(queue_delay,10), 0.15*rtt_min)` | ✅ khớp |
| Eq. 9 | `U_rate_i = B_i` | `rate_utilization` dùng trực tiếp làm `U_rate` | ✅ khớp (không tách biến riêng) |
| Eq. 10 | `U_delay_i = min(qd_i/q_ref_i, 1)` | `delay_utilization = min(queue_delay/queue_ref, 1)` | ✅ khớp |
| Eq. 11 | `U_i = max(U_rate_i, U_delay_i)` | `utilization = max(rate_utilization, delay_utilization)` | ✅ khớp |
| Eq. 12 | `l_i = max(G_i - 1, 0)` | dùng trực tiếp `max(gain-1.0, 0.0)` trong reward | ✅ khớp (không tách biến riêng) |
| Eq. 13 | `r_i = B_i - λ1·C_i - λ2·U_i·l_i` (reward "tại chỗ" tổng quát) | **không tồn tại như một hàm riêng** | ⚠️ **không triển khai độc lập** — bị Eq. 16 (mục A5) thay thế làm định nghĩa reward duy nhất trong repo |

### A4. Action space & phase mask (Eq. 14–15)

**Paper:**
```
A = {0.90,0.92,0.94,0.96,0.98,1.00,1.05,1.10,1.15,1.20,1.25}         (Eq. 14)
A_UP = {1.05,1.10,1.15,1.20,1.25}, A_DOWN = {0.90,...,0.98}, A_CRUISE={1.00}  (Eq. 15)
```
**Code:** `utils/bbr.py`
```python
PACING_GAINS = (0.90, 0.92, 0.94, 0.96, 0.98, 1.00, 1.05, 1.10, 1.15, 1.20, 1.25)
PHASE_ACTION_INDICES = {BW_DOWN: (0,1,2,3,4), BW_CRUISE: (5,), BW_UP: (6,7,8,9,10)}
```
**Kết quả:** ✅ khớp tuyệt đối, từng con số một. Đây là "single source of truth" dùng chung cho preprocessing, model head, và mask logits khi suy luận (`mask_action_logits`, `mask_sequence_logits`) lẫn khi tính loss (phần B).

### A5. Reward tối ưu & nhãn chuyên gia (Eq. 16–18) — ⚠️ **khác biệt quan trọng nhất**

**Paper (Algorithm 1, dòng 15–19 + Eq. 16–18):**
```
for mỗi g ∈ A_φi:
    ước lượng B̂_i(g), τ̂_i(g)                       # "trace-driven model"
    R(s_i,g) = B̂_i(g)/B_ref_i − λ1·τ̂_i(g)/τ_ref_i − λ2·U_i·max(g−1,0)   (Eq. 16)
a*_i = argmax_{g∈A_φi} R(s_i,g)                                          (Eq. 17)
D_BBR = {(s_i, a*_i, R(s_i,a*_i))}                                       (Eq. 18)
```
Tức là: **nhãn chuyên gia `a*_i` phải là hành động tối ưu hoá reward**, duyệt qua *toàn bộ* action khả dĩ trong phase đó.

**Code (`build_paper_labels()`):**
```python
action_index = _equation_2_3_action(phase, utilization_proxy[index])   # chọn nhãn theo Eq. 2-3 (nearest-gain)
reward = _candidate_reward(index, action_index, ...)                   # rồi MỚI tính R() cho đúng 1 action đã chọn
```
Repo **không** duyệt `argmax` qua tập `A_φi` như Eq. 17. Thay vào đó:
1. Nhãn `a*_i` được chọn bằng **Eq. 2–3** (nearest phase-safe gain của ước lượng liên tục).
2. **Eq. 16** chỉ được dùng để tính giá trị `reward` lưu kèm nhãn đó (không dùng để chọn nhãn).

Đây **không phải lỗi ngẫu nhiên** — nó được ghi lại tường minh và có phê duyệt của bạn trong `ADR-0001`:
> *"...literal Equation 16 optimization combined with Equations 22-27 selected only three actions: 0.98, 1.00, and 1.05. That collapse is why the approved exploratory reproduction now treats Equations 2-3 as the authoritative expert-label construction and keeps Equation 16 as the reward calculation for the selected label."*

Lý do: nếu chọn nhãn bằng `argmax_g R(s_i,g)` đúng như Eq. 17, phần lớn action sẽ bị loại vì `R(·)` luôn phạt các gain > 1 hoặc < 1 nặng hơn — thực nghiệm cho thấy nó **suy biến (collapse)** chỉ còn 3/11 action được chọn trong toàn bộ tập dữ liệu. Vì vậy bạn đã quyết định (có ghi log ngày 2026-08-06) dùng Eq. 2–3 làm luật chọn nhãn chính thức, Eq. 16 chỉ là "điểm thưởng" đi kèm — **không phải argmax reward optimizer** như bản chất Eq. 17 mô tả.

**Kết luận A5:** ⚠️ **deviation có chủ đích, đã được phê duyệt** — không khớp Eq. 17 theo nghĩa đen, nhưng có tài liệu hoá đầy đủ lý do và tác động (dataset version `slm-bbr-paper-w10-eq2-3-labels-v1-exploratory` phản ánh đúng điều này trong tên).

### A6. Structured sequence sampling (Eq. 19) & LoRA

**Paper Eq. 19:** cửa sổ trượt độ dài `w` lấy mẫu từ `D_BBR`, mỗi mẫu gồm `{R_i, s_t^1..s_t^9, a_t^1..a_t^h}`.

**Code:** `generate_bbr_dev_split.py` / `plm_special/data/dataset.py` — cắt cửa sổ độ dài 20 (`sequence length 20, sample step 20` — khớp `docs/PAPER_DRAFT.md` §III-D), không chồng lấp (`step 20`). ✅ khớp về cấu trúc; paper không cho số cụ thể (w=20) — đây là lựa chọn của bạn (ghi trong `ADR-0002`), không phải trích từ paper.

**LoRA** (paper Section III-C, không có công thức số, chỉ mô tả `W_T = A·B`, `r ≪ min(p,q)`):
Code dùng thư viện `peft` (`plm_special/lora.py :: attach_modern_lora`) — `peft` áp dụng công thức LoRA chuẩn (Hu et al. 2021) có **hệ số scale**:
```
W = W_0 + (α/r)·B·A
```
Đây là công thức tường minh trong `docs/PAPER_DRAFT.md` §III-E (không phải trong paper SLM-BBR gốc — paper gốc chỉ viết `W_T = A·B` không có `α/r`). Vậy: ✅ triển khai đúng chuẩn LoRA phổ biến, chính xác hơn mô tả rút gọn trong paper gốc, và khớp với công thức bạn tự viết lại trong paper của mình.

### A7. Training loss (Eq. 20) — có **hai đường dẫn khác nhau** trong repo

**Paper Eq. 20:**
```
L = (1/w) Σ_{t'=1}^{w} Σ_{j=1}^{m} L_H(a_t'^j, â_t'^j)      (cross-entropy, trung bình theo horizon)
```

Repo có **2 pipeline huấn luyện** với 2 cách tính loss khác nhau:

| Pipeline | File | Cách tính loss | Có mask theo phase không? |
|---|---|---|---|
| Cũ (4-SLM gốc: GPT-2/T5/GPT-Neo/SmolLM2) | `run_plm.py` + `plm_special/trainer.py` | `nn.CrossEntropyLoss()` áp thẳng lên 11 logits | ❌ **không mask** — đây chính là "unmasked training loss" mà `ADR-0012` liệt kê là lỗi (*defect*) chưa sửa của pipeline cũ |
| Mới (dùng cho `gpt_classical`/`gpt_quantum`) | `train_modern_lora.py` + `utils/training_metrics.py :: masked_cross_entropy()` | `mask_sequence_logits()` che action không hợp lệ (`-inf`) **trước khi** cross-entropy | ✅ **có mask** |

Điều thú vị: `plm_special/models/rl_policy.py :: forward()` (dùng khi train) **không** gọi `mask_action_logits`, chỉ `sample()` (dùng khi suy luận/inference) mới mask. Vì vậy pipeline cũ (`run_plm.py`) train trên loss không mask dù action không hợp lệ vẫn được model "nhìn thấy" ở lúc train.

Paper của bạn (`docs/PAPER_DRAFT.md` §III-I) viết công thức loss **đã có mask ngay trong công thức**:
```
L̃ = -(1/L) Σ_t log( exp(õ_{t,a*_t}) / Σ_{j∈A_φt} exp(õ_{t,j}) )
```
→ công thức này khớp chính xác với `masked_cross_entropy()` trong `utils/training_metrics.py` (pipeline mới), **không khớp** với `plm_special/trainer.py` (pipeline cũ, không mask). Đây là điểm quan trọng cần biết nếu bạn trích dẫn công thức loss trong paper: công thức trong paper mô tả đúng pipeline `train_modern_lora.py`, không mô tả `run_plm.py`.

### A8. Surrogate throughput/retransmission model (Eq. 21–27)

| Paper | Công thức | Code (`utils/tokyo_evaluation.py :: paper_surrogate_episode`, và tái sử dụng trong `_candidate_reward` cho việc gán reward nhãn) | Trạng thái |
|---|---|---|---|
| Eq. 21 | `b_send(i) = G(a_i)·b_Bw(i)` | không tính riêng — chỉ là bước dẫn nhập cho Eq. 22 | (mang tính diễn giải, không cần code riêng) |
| Eq. 22 | `T_SLM(i) = B_cap(i)·min(G(a_i), 1.0)` | `predicted_throughput = b_cap * minimum(gains, 1.0)` | ✅ khớp |
| Eq. 23 | `S(G_i) = max(softplus(β(G_i−1)) − softplus(0), 0)` | `strength = max(softplus(β(gain-1)) - softplus(0), 0)` | ✅ khớp (β=5, `Table I`) |
| Eq. 24 | `Φ(G_i) = (S(G_i)/S(G_max))^α` | `phi = (strength/maximum_strength) ** α` | ✅ khớp (α=1.5) |
| Eq. 25 | `L_i = U_i·[ε + (1−ε)·Φ(G_i)]` | `loss_factor = utilization * (ε + (1-ε)*phi)` | ✅ khớp (ε=1e-3) |
| Eq. 26 | `L_i ← L_i·(1 − κ_down·(1−G_i))` khi `G_i<1` | `loss_factor[below_one] *= 1 - κ_down*(1-gains[...])` | ✅ khớp (κ_down=0.5) |
| Eq. 27 | `τ_SLM_i = τ_min + (τ_max,i − τ_min)·L_i` | `predicted_retransmits = tau_min + (tau_max-tau_min)*loss_factor` | ✅ khớp |

**Kết quả:** ✅ Eq. 22–27 được triển khai **rất trung thành**, dùng ở đúng 2 nơi: (1) `utils/tokyo_evaluation.py` — đánh giá hậu-huấn-luyện trên Tokyo (đúng mục đích ban đầu của paper, Section IV-B/C); và (2) `utils/starlink_preprocessing.py :: _candidate_reward` — dùng lại y hệt công thức này để đóng vai trò "trace-driven model" `B̂_i(g)`, `τ̂_i(g)` mà Algorithm 1 dòng 16 nhắc tới nhưng không định nghĩa cụ thể. Việc tái sử dụng Eq. 22–27 làm surrogate cho Eq. 16 là một **quyết định thiết kế của bạn**, được `ADR-0001` ghi nhận minh bạch ("Equation 16, backed by the throughput and retransmission surrogates from Equations 22-27").

---

## PHẦN B — Paper của bạn (`docs/PAPER_DRAFT.md`) ↔ `plm_special/quantum_head.py`, `classical_head.py`

### B1. Classical head (§III-G)

**Paper (bạn viết):**
```
o_t^classical = W_c·h_t + b_c ∈ R^11
```
**Code:** `plm_special/models/rl_policy.py`
```python
self.action_head = nn.Linear(plm_embed_size, action_levels)   # head_type == "classical"
```
✅ khớp chính xác — một lớp Linear thuần, không có bottleneck. Đây chính là head dùng cho `gpt_classical` (kết quả đã "frozen" trong paper).

### B2. Quantum head (§III-H)

**Paper (bạn viết):**
```
θ_t = s_θ · tanh( (W_q·LayerNorm(h_t) + b_q) / T )
→ mỗi θ_{t,i} encode bằng R_Y(θ_{t,i}) trên qubit i
→ depth-D: mỗi layer có R_Y(ω_{d,i}) huấn luyện được + CNOT ring entangle
→ đo Pauli-Z mỗi qubit: q_t ∈ [-1,1]^{n_q}
o_t^quantum = W_out·q_t + b_out
```
**Code:** `plm_special/quantum_head.py :: QuantumActionHead.forward()`
```python
projected = self.angle_projection(self.input_norm(hidden))   # W_q·LayerNorm(h_t) + b_q
bounded   = torch.tanh(projected / self.temperature)          # tanh(.../T)
angles    = self.angle_scale * bounded                        # s_θ · tanh(...) = θ_t
expectations = self.quantum_layer(angles...)                  # Qiskit EstimatorQNN, RY encode + RY(ω) + CNOT ring, đo Pauli-Z → q_t
logits = self.output_projection(expectations)                  # W_out·q_t + b_out
```
✅ khớp **từng bước một, 1:1**, kể cả thứ tự LayerNorm → Linear → chia nhiệt độ `T` → tanh → nhân `angle_scale`. Điều này hợp lý vì công thức trong `PAPER_DRAFT.md` được viết *mô tả lại đúng* những gì `quantum_head.py` đã làm (paper của bạn viết sau khi code đã tồn tại), nên không có gì đáng ngạc nhiên khi chúng khớp tuyệt đối — khác với Phần A (paper SLM-BBR viết trước, code phải tự diễn giải chi tiết còn thiếu).

Cấu hình "đóng băng" cho `gpt_quantum` mà paper nêu (`8 qubits, depth 1, RY-only, angle_scale=π, LayerNorm bật, T=4`) khớp với tham số mặc định truyền vào `QuantumActionHead(n_qubits=8, depth=1, ansatz="trainable_ry_layers", input_layernorm=True, temperature=4.0, angle_scale="pi")` khi khởi tạo trong script train (không paste lại toàn bộ vì đây là cấu hình chạy, không phải công thức).

### B3. Phase-masked loss (§III-I)

**Paper (bạn viết):**
```
õ_{t,j} = o_{t,j} nếu j ∈ A_φt, else -∞
L = -(1/L) Σ_t log( exp(õ_{t,a*_t}) / Σ_{j∈A_φt} exp(õ_{t,j}) )
```
**Code:** `utils/training_metrics.py :: masked_cross_entropy()`
```python
masked_logits = mask_sequence_logits(logits, phases)     # õ_{t,j}, -inf ngoài A_φt
... F.cross_entropy(flat_logits, labels) ...              # = -log(softmax) đúng công thức trên
```
✅ khớp — như đã nói ở A7, công thức này mô tả đúng pipeline **mới** (`train_modern_lora.py`), không phải pipeline cũ (`run_plm.py`/`plm_special/trainer.py`, không mask).

---

## Bảng tổng hợp nhanh

| # | Công thức | Nguồn | Trạng thái trong code |
|---|---|---|---|
| Eq.1 | Deviation phase detector | SLM-BBR | ✅ khớp |
| Eq.2–3 | Continuous gain + nearest discretisation | SLM-BBR | ✅ khớp, dùng làm **luật chọn nhãn chính thức** |
| Eq.4 | `B_i` | SLM-BBR | ✅ khớp (ẩn trong biến `rate_utilization`) |
| Eq.5 | `τ̄_i = tanh(...)` | SLM-BBR | ❌ không triển khai (bị Eq.27 thay thế) |
| Eq.6–7 | `B_ref`, `τ_ref` | SLM-BBR | ✅ khớp |
| Eq.8–11 | `q_ref`, `U_rate`, `U_delay`, `U_i` | SLM-BBR | ✅ khớp |
| Eq.12 | `l_i` | SLM-BBR | ✅ khớp (inline) |
| Eq.13 | reward rút gọn `r_i` | SLM-BBR | ⚠️ không triển khai độc lập, bị Eq.16 thay thế |
| Eq.14–15 | Action space + phase mask | SLM-BBR | ✅ khớp tuyệt đối |
| Eq.16 | `R(s_i,g)` | SLM-BBR | ✅ khớp về công thức, nhưng chỉ tính cho **1 action đã chọn sẵn**, không quét toàn bộ `A_φi` |
| Eq.17 | `a*_i = argmax_g R(s_i,g)` | SLM-BBR | ❌ **không triển khai theo nghĩa đen** — nhãn lấy từ Eq.2–3 thay vì argmax (quyết định có chủ đích, xem `ADR-0001`) |
| Eq.18 | `D_BBR` | SLM-BBR | ✅ khớp cấu trúc |
| Eq.19 | Cửa sổ trượt | SLM-BBR | ✅ khớp cấu trúc (w=20 là lựa chọn riêng) |
| Eq.20 | Cross-entropy loss | SLM-BBR | ⚠️ khớp trong `train_modern_lora.py`, **không mask** trong `run_plm.py`/`trainer.py` cũ |
| Eq.21 | `b_send` | SLM-BBR | (chỉ mang tính diễn giải, không cần code) |
| Eq.22–27 | Surrogate throughput/retransmission | SLM-BBR | ✅ khớp rất trung thành, dùng lại cho cả Tokyo eval lẫn Eq.16 |
| LoRA `W=W_0+(α/r)BA` | | Paper của bạn / peft | ✅ khớp (paper SLM-BBR gốc chỉ viết `W=AB`, không có hệ số scale) |
| `o^classical = W_c h_t+b_c` | | Paper của bạn | ✅ khớp `nn.Linear` |
| `θ_t`, `o^quantum` | | Paper của bạn | ✅ khớp `quantum_head.py` từng bước |
| Phase-masked CE | | Paper của bạn | ✅ khớp `masked_cross_entropy()` (chỉ pipeline mới) |

## Ba điểm cần chú ý nhất nếu bạn viết lại phần Methodology

1. **Eq.17 (argmax reward) không được dùng để chọn nhãn** — nhãn chuyên gia thực tế dùng Eq.2–3 (nearest-gain), Eq.16 chỉ gắn nhãn với một điểm reward. Nếu paper của bạn mô tả quy trình gán nhãn, nên nói rõ đây là "Eq.2–3 nearest discretisation, reward-scored via Eq.16 surrogate" chứ không phải "reward-argmax selection" — đúng như bạn đã ghi trong `ADR-0001`, chỉ cần đảm bảo `PAPER_DRAFT.md` diễn đạt nhất quán với điều này.
2. **Hai loss khác nhau tồn tại song song trong repo** — nếu số liệu `gpt_classical`/`gpt_quantum` trong paper đến từ `train_modern_lora.py`, công thức phase-masked CE trong `PAPER_DRAFT.md` mô tả đúng. Nhưng nếu có ai đối chiếu với `run_plm.py` (pipeline 4-SLM cũ), sẽ thấy loss đó không mask — nên nêu rõ trong phần Limitations/Reproducibility nếu cả hai pipeline được nhắc tới trong cùng bài báo.
3. **Eq.5 không tồn tại trong code** — không sai (vì Eq.13/Eq.5 bị Eq.16 thay thế toàn bộ), nhưng nếu ai đó review bám sát từng công thức của paper gốc, nên có một câu giải thích ngắn (như `ADR-0001` đã làm) để không bị coi là "thiếu sót".

---
*Nguồn: `project_sources/2607.07142v1 copy.pdf` (paper SLM-BBR gốc), `docs/PAPER_DRAFT.md`, `utils/starlink_preprocessing.py`, `utils/bbr.py`, `utils/tokyo_evaluation.py`, `plm_special/quantum_head.py`, `plm_special/classical_head.py`, `plm_special/models/rl_policy.py`, `plm_special/trainer.py`, `utils/training_metrics.py`, `plm_special/lora.py`, `ADR-0001`, `ADR-0012`.*
