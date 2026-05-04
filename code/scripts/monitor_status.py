import os
import pandas as pd
import time
from datetime import timedelta

def get_last_line_info(file_path):
    """Reads the last line of a file efficiently."""
    try:
        with open(file_path, 'rb') as f:
            f.seek(0, os.SEEK_END)
            if f.tell() == 0:
                return None
            
            # Go backwards to find the newline
            f.seek(-2, os.SEEK_CUR)
            while f.read(1) != b'\n':
                f.seek(-2, os.SEEK_CUR)
                if f.tell() == 0:
                    break
            
            last_line = f.readline().decode().strip()
            return last_line.split('\t')
    except Exception:
        return None

def monitor_training_realtime():
    base_dir = "catboost_info"
    time_left_path = os.path.join(base_dir, "time_left.tsv")
    test_error_path = os.path.join(base_dir, "test_error.tsv")

    print(f"🚀 **CatBoost 실시간 모니터링 시작** (종료하려면 Ctrl+C)")
    print(f"{'Iter':<10} {'Passed':<12} {'Remaining':<12} {'Latest MAE':<15} {'Best MAE':<15}")
    print("-" * 65)

    last_iter = -1
    best_mae = float('inf')

    try:
        while True:
            if not os.path.exists(time_left_path):
                time.sleep(2)
                continue

            # Read time_left.tsv
            time_row = get_last_line_info(time_left_path)
            if not time_row or len(time_row) < 3:
                time.sleep(1)
                continue
            
            try:
                # time_left header: iter, Passed, Remaining
                current_iter = int(time_row[0])
                passed = str(timedelta(seconds=int(float(time_row[1]))))
                remaining = str(timedelta(seconds=int(float(time_row[2]))))
            except ValueError:
                time.sleep(1)
                continue

            if current_iter > last_iter:
                # Read test_error.tsv
                current_mae = "N/A"
                if os.path.exists(test_error_path):
                    error_row = get_last_line_info(test_error_path)
                    if error_row and len(error_row) >= 2:
                        try:
                            val = float(error_row[1])
                            current_mae = f"{val:.5f}"
                            if val < best_mae:
                                best_mae = val
                        except ValueError:
                            pass
                
                best_mae_str = f"{best_mae:.5f}" if best_mae != float('inf') else "N/A"

                print(f"{current_iter:<10} {passed:<12} {remaining:<12} {current_mae:<15} {best_mae_str:<15}")
                last_iter = current_iter
            
            time.sleep(2)

    except KeyboardInterrupt:
        print("\n🛑 모니터링 종료")
    except Exception as e:
        print(f"\n⚠️ 에러 발생: {e}")

if __name__ == "__main__":
    monitor_training_realtime()
