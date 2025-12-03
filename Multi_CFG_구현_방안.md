# Multi-Objective CFG 구현 방안 (응집성 지표)

## 참고 논문 및 방법론

### 1. InstructPix2Pix (Dual-CFG) [참고]
- **핵심**: 각 조건에 독립적인 guidance scale 부여
- **공식**: $\hat{\epsilon}_{cfg} = \epsilon_b + \text{cfg}_r \cdot (\epsilon_r - \epsilon_b) + \text{cfg}_a \cdot (\epsilon_a - \epsilon_r)$
- **적용**: 각 응집성 지표마다 독립적인 guidance scale 사용

### 2. GCDM (Generalized Composable Diffusion Models) [참고]
- **핵심**: Joint guidance와 independent guidance의 가중 결합
- **공식**: $\nabla_x \log p(x|z_c, z_s) = \nabla_x \log p(x) + \alpha \Big[ \lambda \nabla_x \log p(z_c, z_s | x) + (1-\lambda) \big( \beta_c \nabla_x \log p(z_c|x) + \beta_s \nabla_x \log p(z_s|x) \big) \Big]$
- **적용**: 응집성 지표 간 상관관계를 고려한 joint guidance 추가

### 3. PROUD (Pareto-Guided Diffusion) [참고]
- **핵심**: Pareto optimality 기반 동적 가중치 결정
- **적용**: 여러 응집성 지표 간 trade-off를 Pareto front로 최적화

### 4. MUDM (Multi-objective Unconditional Diffusion Model) [참고]
- **핵심**: Property 간 dependency graph 정의
- **적용**: 응집성 지표 간 상관관계를 그래프로 모델링

---

## 구현 방안: 하이브리드 접근법

### Phase 1: Independent Multi-CFG (InstructPix2Pix 방식) ⭐⭐⭐⭐⭐

**이유:**
- 구현이 가장 간단하고 직관적
- 각 지표를 독립적으로 제어 가능
- Trade-off 분석이 용이

**구현 방법:**

```python
# diffab/models/diffab.py

class DiffusionAntibodyDesign(nn.Module):
    def __init__(self, cfg):
        # ... 기존 코드 ...
        
        # Multi-objective CFG: 각 응집성 지표별 독립 embedding
        self.cond_dims = {
            'gravy': cfg.condition_dim // 6,
            'net_charge': cfg.condition_dim // 6,
            'pI_distance': cfg.condition_dim // 6,
            'aromaticity': cfg.condition_dim // 6,
            'aliphatic': cfg.condition_dim // 6,
            'positive': cfg.condition_dim // 6,
        }
        
        # 각 지표별 embedding network
        self.gravy_embed = nn.Sequential(
            nn.Linear(1, self.cond_dims['gravy']*2), nn.ReLU(),
            nn.Linear(self.cond_dims['gravy']*2, self.cond_dims['gravy'])
        )
        self.net_charge_embed = nn.Sequential(...)
        self.pI_distance_embed = nn.Sequential(...)
        self.aromaticity_embed = nn.Sequential(...)
        self.aliphatic_embed = nn.Sequential(...)
        self.positive_embed = nn.Sequential(...)
        
        # Unconditional embedding
        self.uncond_embed = nn.Parameter(torch.randn(cfg.condition_dim))
        
        # 각 지표별 guidance scale (학습 가능하거나 하이퍼파라미터)
        self.guidance_scales = {
            'gravy': cfg.get('guidance_scale_gravy', 1.0),
            'net_charge': cfg.get('guidance_scale_net_charge', 1.0),
            'pI_distance': cfg.get('guidance_scale_pI_distance', 1.0),
            'aromaticity': cfg.get('guidance_scale_aromaticity', 1.0),
            'aliphatic': cfg.get('guidance_scale_aliphatic', 1.0),
            'positive': cfg.get('guidance_scale_positive', 1.0),
        }
    
    def _get_y_embed(self, batch, N, L, device):
        """Multi-objective condition embedding (InstructPix2Pix 방식)"""
        embeds = []
        
        # 각 지표별로 embedding 생성
        if 'y_cdr_gravy' in batch:
            gravy_embed = self.gravy_embed(batch['y_cdr_gravy'].view(-1, 1))
            gravy_embed = gravy_embed.unsqueeze(1).expand(N, L, -1)
            embeds.append(gravy_embed)
        else:
            embeds.append(torch.zeros(N, L, self.cond_dims['gravy'], device=device))
        
        if 'y_cdr_net_charge' in batch:
            net_charge_embed = self.net_charge_embed(batch['y_cdr_net_charge'].view(-1, 1))
            net_charge_embed = net_charge_embed.unsqueeze(1).expand(N, L, -1)
            embeds.append(net_charge_embed)
        else:
            embeds.append(torch.zeros(N, L, self.cond_dims['net_charge'], device=device))
        
        # ... 나머지 지표도 동일하게 ...
        
        y_embed = torch.cat(embeds, dim=-1)  # (N, L, cond_dim)
        
        # CFG dropout (10% unconditional)
        drop_mask = (torch.rand(N, 1, device=device) < self.y_drop_prob)
        uncond_embed = self.uncond_embed.unsqueeze(0).expand(N, L, -1)
        drop_mask = drop_mask.unsqueeze(1).expand(-1, L, self.cond_dim)
        y_embed = torch.where(drop_mask, uncond_embed, y_embed)
        
        return y_embed
```

**Sampling 시 각 지표별 독립 guidance:**

```python
# diffab/modules/diffusion/dpm_full.py

class FullDPM(nn.Module):
    @torch.no_grad()
    def sample(self, ..., y_targets=None, guidance_scales=None):
        """
        Args:
            y_targets: dict with keys ['gravy', 'net_charge', ...]
            guidance_scales: dict with guidance scale for each metric
        """
        # ... 기존 코드 ...
        
        for t in pbar(range(self.num_steps, 0, -1)):
            # Unconditional prediction
            v_next_u, R_next_u, eps_p_u, c_denoised_u = self.eps_net(
                v_t, p_t, s_t, res_feat, pair_feat, beta, 
                mask_generate, mask_res, y_embed_uncond
            )
            
            if y_targets is not None:
                # 각 지표별 conditional prediction
                v_next_cond = v_next_u.clone()
                eps_p_cond = eps_p_u.clone()
                c_denoised_cond = c_denoised_u.clone()
                
                # InstructPix2Pix 방식: 각 조건을 순차적으로 적용
                for metric_name in ['gravy', 'net_charge', 'pI_distance', 
                                   'aromaticity', 'aliphatic', 'positive']:
                    if metric_name in y_targets and metric_name in guidance_scales:
                        # 해당 지표만 포함한 y_embed 생성
                        y_embed_single = self._get_single_metric_embed(
                            metric_name, y_targets[metric_name], N, L, device
                        )
                        
                        # Conditional prediction
                        v_next_c, R_next_c, eps_p_c, c_denoised_c = self.eps_net(
                            v_t, p_t, s_t, res_feat, pair_feat, beta,
                            mask_generate, mask_res, y_embed_single
                        )
                        
                        # Guidance 적용 (InstructPix2Pix 방식)
                        cfg_scale = guidance_scales[metric_name]
                        v_next_cond = v_next_cond + cfg_scale * (v_next_c - v_next_u)
                        eps_p_cond = eps_p_cond + cfg_scale * (eps_p_c - eps_p_u)
                        c_denoised_cond = c_denoised_cond + cfg_scale * (c_denoised_c - c_denoised_u)
                
                v_next = v_next_cond
                eps_p = eps_p_cond
                c_denoised = c_denoised_cond
            else:
                v_next = v_next_u
                eps_p = eps_p_u
                c_denoised = c_denoised_u
            
            # ... 나머지 denoising ...
```

**장점:**
- ✅ 구현 간단
- ✅ 각 지표를 독립적으로 제어
- ✅ Trade-off 분석 용이 (각 guidance scale 조절)

**단점:**
- ⚠️ 지표 간 상관관계 미반영
- ⚠️ 순차 적용 시 순서 의존성 가능

---

### Phase 2: Joint + Independent Hybrid (GCDM 방식) ⭐⭐⭐⭐

**이유:**
- 응집성 지표 간 상관관계 반영
- Joint guidance로 더 일관된 생성

**구현 방법:**

```python
class DiffusionAntibodyDesign(nn.Module):
    def __init__(self, cfg):
        # ... Phase 1 코드 ...
        
        # Joint guidance를 위한 추가 embedding
        # 모든 지표를 함께 고려하는 joint embedding
        self.joint_embed = nn.Sequential(
            nn.Linear(6, cfg.condition_dim * 2),  # 6개 지표 입력
            nn.ReLU(),
            nn.Linear(cfg.condition_dim * 2, cfg.condition_dim)
        )
        
        # Joint vs Independent 가중치 (GCDM의 λ)
        self.lambda_joint = cfg.get('lambda_joint', 0.5)  # 0=independent, 1=joint
    
    def _get_y_embed(self, batch, N, L, device):
        """GCDM 방식: Joint + Independent hybrid"""
        # Independent embeddings (Phase 1과 동일)
        indep_embeds = []
        # ... 각 지표별 embedding 생성 ...
        y_embed_indep = torch.cat(indep_embeds, dim=-1)
        
        # Joint embedding (모든 지표를 함께)
        if all(f'y_cdr_{m}' in batch for m in ['gravy', 'net_charge', 'pI_distance', 
                                                'aromaticity', 'aliphatic', 'positive']):
            joint_input = torch.stack([
                batch['y_cdr_gravy'],
                batch['y_cdr_net_charge'],
                batch['y_cdr_pI_distance'],
                batch['y_cdr_aromaticity'],
                batch['y_cdr_aliphatic'],
                batch['y_cdr_positive']
            ], dim=-1)  # (N, 6)
            y_embed_joint = self.joint_embed(joint_input)  # (N, cond_dim)
            y_embed_joint = y_embed_joint.unsqueeze(1).expand(N, L, -1)
        else:
            y_embed_joint = torch.zeros(N, L, self.cond_dim, device=device)
        
        # GCDM 방식: 가중 결합
        y_embed = (self.lambda_joint * y_embed_joint + 
                  (1 - self.lambda_joint) * y_embed_indep)
        
        # CFG dropout
        # ... 기존 코드 ...
        return y_embed
```

**장점:**
- ✅ 지표 간 상관관계 반영
- ✅ Joint guidance로 일관성 향상
- ✅ λ로 조절 가능

**단점:**
- ⚠️ 구현 복잡도 증가
- ⚠️ λ 튜닝 필요

---

### Phase 3: Pareto-Guided (PROUD 방식) ⭐⭐⭐

**이유:**
- 여러 응집성 지표 간 trade-off를 Pareto optimality로 최적화
- 동적 가중치 결정

**구현 방법:**

```python
class ParetoGuidedCFG(nn.Module):
    """
    PROUD 방식: Pareto optimality 기반 동적 가중치
    """
    def __init__(self, num_metrics=6):
        super().__init__()
        self.num_metrics = num_metrics
        
        # Pareto weight predictor
        self.weight_predictor = nn.Sequential(
            nn.Linear(num_metrics, 64),
            nn.ReLU(),
            nn.Linear(64, num_metrics),
            nn.Softmax(dim=-1)  # 가중치 합 = 1
        )
    
    def compute_pareto_weights(self, target_metrics, current_metrics):
        """
        Args:
            target_metrics: (N, 6) - 목표 응집성 지표
            current_metrics: (N, 6) - 현재 예측 지표
        Returns:
            weights: (N, 6) - Pareto optimal 가중치
        """
        # 목표와 현재의 차이
        metric_diff = target_metrics - current_metrics  # (N, 6)
        
        # Pareto weight 계산
        weights = self.weight_predictor(metric_diff)  # (N, 6)
        
        return weights
    
    def forward(self, eps_uncond, eps_conds, weights):
        """
        Args:
            eps_uncond: (N, L, ...) - unconditional prediction
            eps_conds: list of (N, L, ...) - 각 지표별 conditional prediction
            weights: (N, 6) - Pareto weights
        Returns:
            eps_guided: (N, L, ...) - Pareto-guided prediction
        """
        eps_guided = eps_uncond.clone()
        
        for i, eps_cond in enumerate(eps_conds):
            # 각 지표별로 가중치 적용
            w = weights[:, i].unsqueeze(-1).unsqueeze(-1)  # (N, 1, 1)
            eps_guided = eps_guided + w * (eps_cond - eps_uncond)
        
        return eps_guided
```

**장점:**
- ✅ Pareto optimal trade-off
- ✅ 동적 가중치 결정
- ✅ 다중 목표 최적화에 적합

**단점:**
- ⚠️ 구현 복잡도 매우 높음
- ⚠️ 추가 모델 학습 필요
- ⚠️ 계산 비용 증가

---

## 추천 구현 전략

### 단계별 적용

**Phase 1 (즉시 구현)**: Independent Multi-CFG (InstructPix2Pix)
- 가장 간단하고 직관적
- 각 지표를 독립적으로 제어 가능
- Trade-off 분석 용이

**Phase 2 (필요 시)**: Joint + Independent Hybrid (GCDM)
- 지표 간 상관관계가 중요할 때
- 더 일관된 생성이 필요할 때

**Phase 3 (선택적)**: Pareto-Guided (PROUD)
- 정교한 trade-off 최적화가 필요할 때
- 계산 자원이 충분할 때

---

## Trade-off 평가 방법

### 1. Pareto Front 분석 [PROUD 참고]

```python
def compute_pareto_front(samples, metrics):
    """
    여러 샘플의 응집성 지표를 Pareto front로 시각화
    
    Args:
        samples: list of generated structures
        metrics: dict with metric values for each sample
    """
    # 6개 지표를 2D/3D로 projection하여 Pareto front 시각화
    # Hypervolume 계산
    pass
```

**평가지표:**
- **Hypervolume (HV)**: Pareto front가 차지하는 공간 부피 (높을수록 좋음)
- **Pareto Front Coverage**: 목표 영역을 얼마나 커버하는지

### 2. Quality-Diversity Plot [GCDM 참고]

```python
def evaluate_quality_diversity(samples, guidance_scales):
    """
    Guidance scale 조합에 따른 품질-다양성 trade-off
    
    Args:
        samples: generated samples for different guidance scale combinations
        guidance_scales: list of guidance scale dicts
    """
    # FID (quality) vs LPIPS (diversity) plot
    # 또는 응집성 지표 vs 구조 품질 (RMSD) plot
    pass
```

**평가지표:**
- **FID**: 생성 품질 (낮을수록 좋음)
- **LPIPS / Diversity**: 다양성 (높을수록 좋음)
- **Trade-off Curve**: 두 지표 간 곡선

### 3. 개별 지표 MAE [MUDM 참고]

```python
def compute_metric_mae(predicted_metrics, target_metrics):
    """
    각 응집성 지표별 Mean Absolute Error
    
    Args:
        predicted_metrics: (N, 6) - 생성된 구조의 지표
        target_metrics: (N, 6) - 목표 지표
    """
    mae = {}
    metric_names = ['gravy', 'net_charge', 'pI_distance', 
                   'aromaticity', 'aliphatic', 'positive']
    
    for i, name in enumerate(metric_names):
        mae[name] = F.l1_loss(predicted_metrics[:, i], target_metrics[:, i])
    
    return mae
```

**평가지표:**
- **MAE per metric**: 각 지표별 오차
- **Weighted MAE**: 중요도에 따른 가중 평균

### 4. Antibody 특화 지표

```python
def evaluate_antibody_quality(samples, reference):
    """
    Antibody 특화 평가 지표
    
    Args:
        samples: generated antibody structures
        reference: reference structures
    """
    metrics = {
        'rmsd': compute_rmsd(samples, reference),
        'aar': compute_amino_acid_recovery(samples, reference),
        'binding_energy': compute_binding_energy(samples),
        'cohesiveness': compute_cohesiveness_metrics(samples),
    }
    return metrics
```

**평가지표:**
- **RMSD**: 구조 품질
- **AAR**: 서열 복원율
- **Binding Energy**: 결합 에너지
- **Cohesiveness Metrics**: 응집성 지표

---

## 구체적 구현 예시

### 데이터 전처리

```python
# diffab/datasets/sabdab.py

def compute_cdr_cohesiveness_metrics(aa, cdr_flag, mask_residue):
    """CDR 영역만 추출하여 응집성 지표 계산"""
    # ... 구현 (CFG_응집성_지표_설계.md 참고) ...
    return {
        'gravy': ...,
        'net_charge': ...,
        'pI_distance': ...,
        'aromaticity': ...,
        'aliphatic_index': ...,
        'positive_fraction': ...
    }

def preprocess_sabdab_structure(task):
    # ... 기존 전처리 ...
    
    # CDR 응집성 지표 계산
    if structure['heavy'] is not None:
        cdr_metrics = compute_cdr_cohesiveness_metrics(
            aa=parsed['aa'],
            cdr_flag=parsed['cdr_flag'],
            mask_residue=parsed['mask_heavyatom'][:, BBHeavyAtom.CA]
        )
        
        parsed['y_cdr_gravy'] = cdr_metrics['gravy']
        parsed['y_cdr_net_charge'] = cdr_metrics['net_charge']
        parsed['y_cdr_pI_distance'] = cdr_metrics['pI_distance']
        parsed['y_cdr_aromaticity'] = cdr_metrics['aromaticity']
        parsed['y_cdr_aliphatic'] = cdr_metrics['aliphatic_index']
        parsed['y_cdr_positive'] = cdr_metrics['positive_fraction']
    
    return parsed
```

### 학습 시 평가

```python
# 학습 루프에서

# 1. 생성 샘플의 응집성 지표 계산
predicted_metrics = compute_cdr_cohesiveness_metrics(
    generated_aa, cdr_flag, mask_residue
)

# 2. Target과 비교
target_metrics = {
    'gravy': batch['y_cdr_gravy'],
    'net_charge': batch['y_cdr_net_charge'],
    # ...
}

# 3. MAE 계산
mae = compute_metric_mae(predicted_metrics, target_metrics)

# 4. Pareto front 업데이트 (optional)
pareto_front.update(predicted_metrics)
```

---

## 요약

### 구현 우선순위

1. **Phase 1**: Independent Multi-CFG (InstructPix2Pix) - 즉시 구현
2. **Phase 2**: Joint + Independent Hybrid (GCDM) - 필요 시 추가
3. **Phase 3**: Pareto-Guided (PROUD) - 선택적

### Trade-off 평가

1. **Pareto Front 분석**: Hypervolume, Coverage
2. **Quality-Diversity Plot**: FID vs LPIPS
3. **개별 지표 MAE**: 각 응집성 지표별 오차
4. **Antibody 특화 지표**: RMSD, AAR, Binding Energy

### 참고 논문

- **InstructPix2Pix**: Independent guidance scales
- **GCDM**: Joint + Independent hybrid
- **PROUD**: Pareto optimality
- **MUDM**: Property dependency graph

