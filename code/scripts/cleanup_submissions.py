import pandas as pd
import re
import os

def parse_submission_log(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path}를 찾을 수 없습니다.")
        
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 정규식 패턴: ID, 파일명, 날짜, 점수 추출
    # 제출기록.txt의 특수한 줄바꿈 구조 반영
    pattern = r'(\d{7})\s*\n([^\n]+\.csv)\n메모 추가\s+(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+([\d\.]+)'
    matches = re.findall(pattern, content)
    
    if not matches:
        # 패턴이 일치하지 않을 경우를 대비한 보조 패턴
        pattern = r'(\d{7})\s*\n([^\n]+\.csv).*?(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+([\d\.]+)'
        matches = re.findall(content, re.DOTALL)

    df = pd.DataFrame(matches, columns=['submission_id', 'filename', 'timestamp', 'public_score'])
    df['public_score'] = pd.to_numeric(df['public_score'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    return df

# 파일 경로 설정
input_file = '제출기록.txt'
output_dir = 'analysis_results'
output_file = os.path.join(output_dir, 'submission_history_cleaned.csv')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

try:
    df = parse_submission_log(input_file)
    df = df.sort_values('timestamp', ascending=False)
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"SUCCESS: {len(df)} items processed.")
    print(f"LOCATION: {output_file}")
    
    if not df.empty:
        best_idx = df['public_score'].idxmin()
        print(f"BEST_SCORE: {df.loc[best_idx, 'public_score']}")
        print(f"BEST_FILE: {df.loc[best_idx, 'filename']}")
    
except Exception as e:
    print(f"ERROR: {e}")
