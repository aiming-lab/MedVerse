import json
import re
import pandas as pd
import networkx as nx

FILE_PATH = '/Users/jimchen/Documents/Med-Cot/codes/datasets/json/Med_Reason_Final_10_3.json'
OUTPUT_PATH = '/Users/jimchen/Documents/Med-Cot/codes/datasets/json/complex_dag_filtered_small.json'

MIN_BRANCH = 3
MIN_MERGE = 3

MAX_NODES = 7

def parse_reason_path_to_graph(path_text):
    """
    将路径文本解析为 NetworkX DiGraph
    适配格式: "1: A->B \n2: C->D"
    """
    G = nx.DiGraph()
    if not isinstance(path_text, str):
        return G
    
    lines = [line.strip() for line in path_text.split('\n') if line.strip()]
    
    for line in lines:
        if line == '###': continue

        clean_line = re.sub(r'^\d+[:.]?\s*', '', line)
        
        nodes = [n.strip() for n in clean_line.split('->') if n.strip()]
        
        for i in range(len(nodes) - 1):
            u, v = nodes[i], nodes[i+1]
            G.add_edge(u, v)
    return G

def analyze_complexity(G):
    """
    计算复杂度
    """
    if len(G.nodes) == 0:
        return 0, 0, 0
    
    n_branch = sum(1 for n, d in G.out_degree() if d > 1)
    n_merge = sum(1 for n, d in G.in_degree() if d > 1)
    n_total = G.number_of_nodes()
    
    return n_branch, n_merge, n_total

def main():
    try:
        print(f"读取文件: {FILE_PATH} ...")
        with open(FILE_PATH, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        print(f"共加载 {len(raw_data)} 条数据。正在分析图结构...")
        
        results = []
        
        for key_id, item_data in raw_data.items():
            path_text = item_data.get('Original Reason Path', '')
            
            G = parse_reason_path_to_graph(path_text)
            
            branch_count, merge_count, node_count = analyze_complexity(G)
            
            record = item_data.copy()
            record['id'] = key_id
            record['metrics_branch'] = branch_count
            record['metrics_merge'] = merge_count
            record['node_count'] = node_count
            record['density_score'] = (branch_count + merge_count) / node_count if node_count > 0 else 0
            
            results.append(record)

        df = pd.DataFrame(results)
        
        complex_df = df[
            (df['metrics_branch'] >= MIN_BRANCH) & 
            (df['metrics_merge'] >= MIN_MERGE) &
            (df['node_count'] <= MAX_NODES) &
            (df['node_count'] > 1)
        ].copy()
        
        complex_df = complex_df.sort_values(by=['density_score', 'metrics_branch'], ascending=[False, False])
        
        print(f"--------------------------------------------------")
        print(f"筛选标准: 分叉>={MIN_BRANCH}, 合并>={MIN_MERGE}, 节点数<={MAX_NODES}")
        print(f"筛选结果: 发现 {len(complex_df)} 条数据")
        
        if not complex_df.empty:
            print("\nTop 3 最紧凑复杂的案例示例:")
            for i in range(min(3, len(complex_df))):
                row = complex_df.iloc[i]
                print(f"ID: {row['id']} | Nodes: {row['node_count']} | Branch: {row['metrics_branch']} | Merge: {row['metrics_merge']}")
                print(f"Path预览: {str(row['Original Reason Path'])[:60].replace(chr(10), ' ')}...") 
                print("-" * 30)

            out_dict = {}
            for _, row in complex_df.iterrows():
                original_data = row.drop(['id', 'metrics_branch', 'metrics_merge', 'node_count', 'density_score']).to_dict()
                out_dict[row['id']] = original_data

            with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
                json.dump(out_dict, f, indent=2, ensure_ascii=False)
                
            print(f"\n结果已保存至: {OUTPUT_PATH}")
        else:
            print("\n未找到符合条件的复杂数据。")
            print("建议：")
            print("1. 检查是否有节点数 <= 7 的数据。")
            print("2. 尝试将 MIN_BRANCH 或 MIN_MERGE 降低为 2 或 1。")

    except FileNotFoundError:
        print(f"错误: 找不到文件 {FILE_PATH}")
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    main()