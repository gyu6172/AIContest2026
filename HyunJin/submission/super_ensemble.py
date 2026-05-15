import pandas as pd
import os
from collections import Counter

def main():
    print("Super Ensemble v2 (2-Stage Majority Vote) 시작...")

    sub_dir = r"c:\Users\shjsh\Desktop\ai contest\HyunJin\submission"
    sub_base_path   = os.path.join(sub_dir, "submission (7).csv")
    sub_tta_path    = os.path.join(sub_dir, "submission_tta (1).csv")
    sub_prompt_path = os.path.join(sub_dir, "submission_prompt_ensemble.csv")

    df_base   = pd.read_csv(sub_base_path,   encoding='utf-8', encoding_errors='replace').set_index('id')
    df_tta    = pd.read_csv(sub_tta_path,    encoding='utf-8', encoding_errors='replace').set_index('id')
    df_prompt = pd.read_csv(sub_prompt_path, encoding='utf-8', encoding_errors='replace').set_index('id')

    results = []
    stats = {
        "agree_all": 0,
        "majority_target": 0,
        "tie_target": 0,
        "value_corrected": 0,
    }

    for q_id in df_base.index:
        pred_base   = df_base.loc[q_id]
        pred_tta    = df_tta.loc[q_id]    if q_id in df_tta.index    else pred_base
        pred_prompt = df_prompt.loc[q_id] if q_id in df_prompt.index else pred_base

        op1, tid1, val1 = str(pred_base['op']),   str(pred_base['target_id']).strip(),   str(pred_base['value'])
        op2, tid2, val2 = str(pred_tta['op']),    str(pred_tta['target_id']).strip(),    str(pred_tta['value'])
        op3, tid3, val3 = str(pred_prompt['op']), str(pred_prompt['target_id']).strip(), str(pred_prompt['value'])

        # Stage 1: (op, target_id) 다수결
        vote_target = [(op1, tid1), (op2, tid2), (op3, tid3)]
        target_counts = Counter(vote_target)
        best_target = target_counts.most_common(1)[0]

        if best_target[1] == 3:
            winner_op, winner_tid = best_target[0]
            stats["agree_all"] += 1
        elif best_target[1] == 2:
            winner_op, winner_tid = best_target[0]
            stats["majority_target"] += 1
        else:
            winner_op, winner_tid = op3, tid3
            stats["tie_target"] += 1

        # Stage 2: value 다수결 (winner_tid와 target_id가 같은 모델들만 대상)
        value_candidates = []
        if tid1 == winner_tid: value_candidates.append(val1)
        if tid2 == winner_tid: value_candidates.append(val2)
        if tid3 == winner_tid: value_candidates.append(val3)

        if len(value_candidates) >= 2:
            val_counts = Counter(value_candidates)
            best_val, best_val_cnt = val_counts.most_common(1)[0]
            # Prompt Ensemble 기본값과 다르고 2표 이상이면 교정된 것으로 간주
            if best_val != val3 and best_val_cnt >= 2:
                stats["value_corrected"] += 1
            winner_val = best_val
        elif len(value_candidates) == 1:
            winner_val = value_candidates[0]
        else:
            winner_val = val3

        results.append({
            "id":        q_id,
            "op":        winner_op,
            "target_id": winner_tid,
            "value":     winner_val,
        })

    final_df = pd.DataFrame(results)
    output_path = os.path.join(sub_dir, "submission_super_ensemble_v2.csv")
    final_df.to_csv(output_path, index=False)

    print("\n" + "="*60)
    print(f"  병합 완료! 총 {len(final_df)} 문항")
    print(f"  [Stage1-target] 만장일치 (3표):    {stats['agree_all']}건")
    print(f"  [Stage1-target] 다수결 승리 (2표): {stats['majority_target']}건")
    print(f"  [Stage1-target] 완전 의견 대립:    {stats['tie_target']}건 (Prompt Ensemble 채택)")
    print(f"  [Stage2-value]  value 다수결 교정: {stats['value_corrected']}건")
    print("="*60)
    print(f"  최종 제출 파일: {output_path}")

if __name__ == "__main__":
    main()
