import sys

def read_sequence(file_path):
    with open(file_path, 'r') as file:
        return file.readline().strip()

def global_alignment(seq1, seq2, match_score=1, gap_cost=1):
    m, n = len(seq1), len(seq2)
    dp = [[0 for _ in range(n+1)] for _ in range(m+1)]
    
    for i in range(m+1):
        dp[i][0] = -i * gap_cost
    for j in range(n+1):
        dp[0][j] = -j * gap_cost
    
    for i in range(1, m+1):
        for j in range(1, n+1):
            match = dp[i-1][j-1] + (match_score if seq1[i-1] == seq2[j-1] else -match_score)
            delete = dp[i-1][j] - gap_cost
            insert = dp[i][j-1] - gap_cost
            dp[i][j] = max(match, delete, insert)
    
    align1, align2 = "", ""
    i, j = m, n
    while i > 0 and j > 0:
        score_current = dp[i][j]
        score_diagonal = dp[i-1][j-1]
        score_up = dp[i][j-1]
        score_left = dp[i-1][j]
        
        if score_current == score_diagonal + (match_score if seq1[i-1] == seq2[j-1] else -match_score):
            align1 += seq1[i-1]
            align2 += seq2[j-1]
            i -= 1
            j -= 1
        elif score_current == score_left - gap_cost:
            align1 += seq1[i-1]
            align2 += '-'
            i -= 1
        elif score_current == score_up - gap_cost:
            align1 += '-'
            align2 += seq2[j-1]
            j -= 1
    
    while i > 0:
        align1 += seq1[i-1]
        align2 += '-'
        i -= 1
    while j > 0:
        align1 += '-'
        align2 += seq2[j-1]
        j -= 1
    
    align1 = align1[::-1]
    align2 = align2[::-1]
    return align1, align2

def main():
    file_path1 = sys.argv[1]
    file_path2 = sys.argv[2]
    seq1 = read_sequence(file_path1)
    seq2 = read_sequence(file_path2)
    align1, align2 = global_alignment(seq1, seq2)
    print(align1)
    print(align2)

if __name__ == "__main__":
    main()
