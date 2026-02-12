"""
환경 체크 스크립트

목적: LSTM v2 학습 가능 여부 확인
- GPU 메모리: 최소 12GB 여유
- RAM: 최소 8GB 여유
- Disk: 최소 5GB 여유
"""

import torch
import psutil
import shutil

print("=" * 80)
print("환경 체크")
print("=" * 80)

# ============================================================================
# 1. GPU 체크
# ============================================================================
print("\n[1] GPU:")
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    allocated = torch.cuda.memory_allocated() / 1e9
    available = total_memory - allocated

    print(f"  ✅ CUDA 사용 가능")
    print(f"  Device: {device_name}")
    print(f"  Total Memory: {total_memory:.1f} GB")
    print(f"  Allocated: {allocated:.1f} GB")
    print(f"  Available: {available:.1f} GB")

    # 검증
    if available < 12:
        print(f"  ⚠️  경고: Available {available:.1f}GB < 12GB")
        print(f"  권장: Batch size 줄이기 또는 다른 프로세스 종료")
    else:
        print(f"  ✅ Available > 12GB 만족!")

    gpu_ok = True
else:
    print(f"  ❌ CUDA 사용 불가")
    print(f"  LSTM v2는 GPU 필수!")
    gpu_ok = False

# ============================================================================
# 2. RAM 체크
# ============================================================================
print("\n[2] RAM:")
ram = psutil.virtual_memory()
total_ram = ram.total / 1e9
available_ram = ram.available / 1e9
used_ram = ram.used / 1e9
percent_ram = ram.percent

print(f"  Total: {total_ram:.1f} GB")
print(f"  Used: {used_ram:.1f} GB ({percent_ram:.1f}%)")
print(f"  Available: {available_ram:.1f} GB")

# 검증
if available_ram < 8:
    print(f"  ⚠️  경고: Available {available_ram:.1f}GB < 8GB")
    print(f"  권장: 다른 프로그램 종료")
else:
    print(f"  ✅ Available > 8GB 만족!")

ram_ok = available_ram >= 6  # 6GB 이상이면 OK (여유 있게)

# ============================================================================
# 3. Disk 체크
# ============================================================================
print("\n[3] Disk:")
disk = shutil.disk_usage('.')
total_disk = disk.total / 1e9
used_disk = disk.used / 1e9
free_disk = disk.free / 1e9
percent_disk = (used_disk / total_disk) * 100

print(f"  Total: {total_disk:.1f} GB")
print(f"  Used: {used_disk:.1f} GB ({percent_disk:.1f}%)")
print(f"  Free: {free_disk:.1f} GB")

# 검증
if free_disk < 5:
    print(f"  ⚠️  경고: Free {free_disk:.1f}GB < 5GB")
    print(f"  권장: 불필요한 파일 삭제")
else:
    print(f"  ✅ Free > 5GB 만족!")

disk_ok = free_disk >= 3  # 3GB 이상이면 OK

# ============================================================================
# 4. Python 패키지 체크
# ============================================================================
print("\n[4] Python 패키지:")

packages = {
    'torch': torch.__version__,
    'numpy': None,
    'pandas': None,
    'sklearn': None
}

try:
    import numpy as np
    packages['numpy'] = np.__version__
except:
    packages['numpy'] = "❌ 미설치"

try:
    import pandas as pd
    packages['pandas'] = pd.__version__
except:
    packages['pandas'] = "❌ 미설치"

try:
    import sklearn
    packages['sklearn'] = sklearn.__version__
except:
    packages['sklearn'] = "❌ 미설치"

for pkg, version in packages.items():
    status = "✅" if version and "❌" not in str(version) else "❌"
    print(f"  {status} {pkg}: {version}")

packages_ok = all(version and "❌" not in str(version) for version in packages.values())

# ============================================================================
# 5. 최종 결과
# ============================================================================
print("\n" + "=" * 80)
print("최종 결과")
print("=" * 80)

checks = {
    "GPU": gpu_ok,
    "RAM": ram_ok,
    "Disk": disk_ok,
    "Packages": packages_ok
}

all_ok = all(checks.values())

for name, ok in checks.items():
    status = "✅" if ok else "❌"
    print(f"  {status} {name}")

print("\n" + "-" * 80)

if all_ok:
    print("✅ 모든 체크 통과! LSTM v2 학습 가능합니다.")
    print("\n다음 단계:")
    print("  1. 전처리 v2 작성")
    print("  2. 모델 v2 작성")
    print("  3. 학습 스크립트 v2 작성")
    print("  4. 빠른 검증 (1 epoch)")
    print("  5. 전체 학습 (3시간)")
else:
    print("⚠️  일부 체크 실패. 위 권장사항 확인하세요.")
    print("\n실패 항목:")
    for name, ok in checks.items():
        if not ok:
            print(f"  ❌ {name}")

print("=" * 80)
