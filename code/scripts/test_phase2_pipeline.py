import sys
sys.path.append('code/utils')

from kleague_pipeline import KLeaguePipeline

pipeline = KLeaguePipeline()

result = pipeline.run_e2e(
    experiment_name='phase2_last_pass',
    code_path='code/models/best/model_domain_features_v3_last_pass.py',
    cv_scores=[15.3857, 15.3869, 15.3583, 14.7690, 14.7425],
    feature_count=25,
    experiment_type='domain_last_pass',
    expected_cv=15.38,
    expected_gap=0.15
)

if result['success']:
    print(f"\n최종 결과: {'✅ 제출 승인' if result['approved'] else '❌ 제출 거부'}")
    print(f"신뢰도: {result['confidence']:.1f}%")
    print(f"로그: {result['log_path']}")
else:
    print(f"\n❌ Pipeline 실패: Stage {result['stage']}")
