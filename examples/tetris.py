import numpy as np
import matplotlib.pyplot as plt
import random

# Tetris-like 문제 해결: max_seq_len과 max_memory로 배치 구성
def tetris_batching_optimized(seq_lens, max_seq_len, max_memory, rows_per_batch=4):
    seq_lens = np.sort(np.array(seq_lens))[::-1]  # 큰 값부터 정렬
    batches = []
    used = np.zeros(len(seq_lens), dtype=bool)  # 사용된 시퀀스 추적
    batch = []
    current_batch_memory = 0
    current_row_seq_len = 0
    current_row_count = 0

    while not np.all(used):  # 모든 시퀀스가 배치될 때까지
        current_batch = []
        current_row_seq_len = 0
        current_row_count = 0

        for i, seq_len in enumerate(seq_lens):
            if not used[i]:
                # 가로 제약 (max_seq_len)을 넘지 않고 세로 제약 (rows_per_batch) 이내일 때 추가
                if current_row_seq_len + seq_len <= max_seq_len and current_row_count < rows_per_batch:
                    current_batch.append((i, seq_len))  # 시퀀스 번호와 함께 추가
                    current_row_seq_len += seq_len
                    current_row_count += 1
                    used[i] = True  # 해당 시퀀스를 사용 처리
                if current_row_count >= rows_per_batch:
                    break  # 4개의 시퀀스가 쌓이면 배치 완료

        batches.append(current_batch)  # 현재 배치를 완료하고 다음 배치로 이동

    return batches

# 배치를 타일처럼 시각적으로 표시 (번호와 색상 추가)
def plot_batches_as_tiles_with_labels(batches, max_seq_len, max_memory):
    fig, ax = plt.subplots(figsize=(12, len(batches) * 2))  # 배치 크기에 따른 그래프 크기 설정
    y_offset = 0
    colors = plt.cm.get_cmap('tab20', len(batches))  # 다양한 색상 사용

    for batch_idx, batch in enumerate(batches):
        x_offset = 0
        for seq_num, seq_len in batch:
            color = colors(seq_num % 20)  # 색상을 랜덤으로 할당
            rect = plt.Rectangle((x_offset, y_offset), seq_len, 1, edgecolor='black', facecolor=color)
            ax.add_patch(rect)
            # 시퀀스 번호 표시
            ax.text(x_offset + seq_len / 2, y_offset + 0.5, f'{seq_num}', horizontalalignment='center', verticalalignment='center', fontsize=8, color='white')
            x_offset += seq_len
        # 배치 번호 텍스트 표시
        ax.text(-100, y_offset + 0.5, f'Batch {batch_idx + 1}', verticalalignment='center', fontsize=12)
        y_offset += 2  # 다음 배치를 아래로 그리기 위한 y-offset

    # 최대 메모리 라인 그리기
    plt.axvline(max_seq_len, color='red', linestyle='--', label=f'Max Sequence Length ({max_seq_len})')

    ax.set_xlim(0, max_seq_len * 1.1)  # 최대 시퀀스 길이보다 조금 넉넉하게 범위 설정
    ax.set_ylim(0, y_offset)
    ax.set_aspect('auto')
    plt.gca().invert_yaxis()  # 배치를 위에서 아래로 그리기
    plt.xlabel('Sequence Length')
    plt.ylabel('Batches')
    plt.title('Batch Layout (Tetris-like) with Labels')
    plt.legend()
    plt.grid(True)
    plt.show()

# seq_len 생성: 다양한 길이의 시퀀스를 많이 생성
random.seed(42)
seq_lens = [random.randint(50, 1500) for _ in range(50)]  # 100개의 시퀀스 길이

# 배치 구성 (Tetris-like 방식으로 최적화)
max_seq_len = 2048  # 가로로 배치할 수 있는 최대 시퀀스 길이
max_memory = 8192  # 배치 당 최대 메모리 크기
batches = tetris_batching_optimized(seq_lens, max_seq_len, max_memory)

# 배치 타일 그래프로 표시 (번호와 색상 추가)
plot_batches_as_tiles_with_labels(batches, max_seq_len, max_memory)
