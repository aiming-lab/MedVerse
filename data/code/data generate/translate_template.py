import os
import json
import re
import httpx
from openai import OpenAI

API_KEY = ""

PATHS = {
    'json_input': '/home/r22user1/clip/augment_integrate_0602_internal.json',
    'json_output': '/home/r22user1/clip/internal_with_report_0613.json',
    'report_root': '/home/r22user1/clip/train_chinese_report',
    'contrast_txt': '/home/r22user1/clip/contrast_internal_test.txt',
    'contrast_out': '/home/r22user1/clip/contrast_internal_test.out',
    'hydro_txt': '/home/r22user1/clip/hydro_sinal_internal_test.txt',
    'hydro_out': '/home/r22user1/clip/hydro_sinal_internal_test.out',
    'tumor_pos_txt': 'tumor_position.txt',
    'tumor_pos_out': '/home/r22user1/clip/tumor_position_internal.txt',
    'shape_txt': '/home/r22user1/clip/shape_internal_test.txt',
    'shape_out': '/home/r22user1/clip/shape_internal_test.out',
}

CLASS_PATH_MAP = {
    'Glioma': 'Gliomas',
    'Glioneuronal and neuronal tumour': 'Glioneuronal and neuronal tumours',
    'Ependymal tumour': 'Ependymal tumours',
    'Choroid plexus tumour': 'Choroid plexus tumours',
    'Embryonal tumour': 'Embryonal tumours',
    'Pineal tumour': 'Pineal tumours',
    'Cranial and paraspinal nerve tumour': 'Cranial and paraspinal nerve tumours',
    'Meningioma': 'Meningioma',
    'Mesenchymal, non-meningothelial tumour': 'Mesenchymal, non-meningothelial tumours',
    'Melanocytic tumour': 'Melanocytic tumours',
    'Hematolymphoid tumour': 'Hematolymphoid tumours',
    'Germ cell tumour': 'Germ cell tumours',
    'Tumors of the sellar region': 'Tumors of the sellar region',
    'Brain Metastase Tumour': 'Metastases'
}

client = OpenAI(
    api_key=API_KEY,
)

class CacheManager:
    def __init__(self):
        self.contrast_cache = self._load_pair(PATHS['contrast_txt'], PATHS['contrast_out'])
        self.hydro_cache = self._load_single(PATHS['hydro_txt']) 
        self.tumor_pos_cache = self._load_single(PATHS['tumor_pos_txt'])
        self.shape_cache = self._load_pair(PATHS['shape_txt'], PATHS['shape_out'])

    def _load_single(self, filepath):
        data = {}
        if not os.path.exists(filepath): return data
        with open(filepath, 'r') as f:
            lines = [l.strip() for l in f.readlines()]
            for i in range(0, len(lines), 2):
                if i+1 < len(lines):
                    data[lines[i]] = lines[i+1]
        return data

    def _load_pair(self, key_file, val_file):
        data = {}
        if not os.path.exists(key_file) or not os.path.exists(val_file): return data
        with open(key_file, 'r') as kf, open(val_file, 'r') as vf:
            keys = [l.strip() for l in kf.readlines()]
            vals = [l.strip() for l in vf.readlines()]
            min_len = min(len(keys), len(vals))
            for i in range(min_len):
                if keys[i]:
                    data[keys[i]] = vals[i]
        return data

    def append_log(self, file_path, content):
        with open(file_path, 'a') as f:
            f.write(content + '\n')

def load_reports_index(root_path):
    index = {}
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if not file.endswith('.txt'): continue
            full_path = os.path.join(root, file)
            try:
                with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = [l.strip() for l in f.readlines()]
                    for i in range(0, len(lines), 3):
                        if i + 2 < len(lines):
                            pat_id = lines[i].split('_')[0]
                            if pat_id not in index:
                                index[pat_id] = []
                            index[pat_id].append({
                                'full_id': lines[i],
                                'finding': lines[i+1],
                                'impression': lines[i+2]
                            })
            except Exception as e:
                print(f"Read Error {full_path}: {e}")
    return index

def find_patient_report(patient_id, report_index, class_name):
    candidates = report_index.get(patient_id.split('_')[0], [])
    if not candidates:
        for key, vals in report_index.items():
            if patient_id in key or key in patient_id:
                candidates.extend(vals)
                break 
    
    if candidates:
        return candidates[0]['finding'], candidates[0]['impression']
    return None, None

def extract_signal_feature(text, signal_type):
    keywords = {
        'T1': ['T1'],
        'T2': ['T2'],
        'FLAIR': ['FLAIR', 'TIR']
    }
    
    target_indices = []
    for kw in keywords[signal_type]:
        idx = text.find(kw)
        if idx != -1:
            target_indices.append(idx)
    
    if not target_indices:
        return ''

    index = target_indices[0]
    
    context_start = max(0, index - 10)
    snippet = text[context_start:index]
    
    is_clean = True
    for other_type, kws in keywords.items():
        if other_type == signal_type: continue
        for kw in kws:
            if kw in snippet: is_clean = False
    
    signal_desc = ''
    if any(p in snippet for p in [',', '，', '。', '.']):
        pass
    elif is_clean:
        if ('长' in snippet or '高' in snippet) and ('短' in snippet or '低' in snippet) or '混杂' in snippet:
            return 'heterogenous signal'
        elif '长' in snippet or '高' in snippet:
            return 'hyperintense' if signal_type != 'T1' else 'hypointense'
        elif '短' in snippet or '低' in snippet:
            return 'hypointense' if signal_type != 'T1' else 'hyperintense'
        elif '等' in snippet:
            return 'isointense'
    start = index
    while start > 0 and text[start-1] not in [',', '，', '。', '.']:
        start -= 1
    end = index
    while end < len(text) - 1 and text[end+1] not in [',', '，', '。', '.', '、']:
        end += 1
    
    snippet = text[start:end+1]
    
    result = []
    # T1
    if signal_type == 'T1':
        if ('长' in snippet or '高' in snippet) and ('短' in snippet or '低' in snippet) or '混杂' in snippet:
            return 'heterogenous signal'
        if '长' in snippet or '高' in snippet: result.append('hypointense')
        if '短' in snippet or '低' in snippet: result.append('hyperintense')
        if '等' in snippet: result.append('isointense')
    else:
        # T2/FLAIR
        if ('长' in snippet or '高' in snippet) and ('短' in snippet or '低' in snippet) or '混杂' in snippet:
            return 'heterogenous signal'
        if '长' in snippet or '高' in snippet: result.append('hyperintense')
        if '短' in snippet or '低' in snippet: result.append('hypointense')
        if '等' in snippet: result.append('isointense')
        
    return '-'.join(result) if result else ''

def call_llm(system_prompt, user_prompt):
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model="gpt-4",
            temperature=0,
        )
        return response.choices[0].message.content.replace('\n', '').strip()
    except Exception as e:
        print(f"LLM Call Error: {e}")
        return ""

def main():
    with open(PATHS['json_input'], 'r') as f:
        data = json.load(f)
    
    cache = CacheManager()
    report_index = load_reports_index(PATHS['report_root'])
    
    results = {}
    cnt_m = 0
    cnt_not_found = 0
    
    total = len(data)

    for idx, (patient, item) in enumerate(data.items()):
        if idx % 50 == 0: print(f"Processing {idx}/{total}...")

        overall_class = item.get('Overall_class')
        if overall_class not in CLASS_PATH_MAP and overall_class != 'Brain Metastase Tumour':
            pass

        finding, impression = find_patient_report(patient, report_index, overall_class)
        
        if not finding:
            cnt_not_found += 1
            continue

        t1_sig = extract_signal_feature(finding, 'T1')
        t2_sig = extract_signal_feature(finding, 'T2')
        fl_sig = extract_signal_feature(finding, 'FLAIR')
        contrast_signal = cache.contrast_cache.get(patient, '')
        if not contrast_signal and '增强' in finding:
            match = re.search(r"[^，。,.]*增强[^，。,.]*", finding)
            snippet = match.group(0) if match else finding
            prompt_sys = "你是一位医学翻译专家。用户输入的句子可能包含多种内容，请你仅提取并翻译与“增强后强化”相关的部分。请将其翻译为专业医学英语表达，如 'After contrast administration, there is a ... enhancement.'。可使用以下形状词汇：分叶状=lobulated，圆形=round，线状=linear，团块状=mass，结节状=nodular，斑片状=patchy，椭圆状=oval，花环状=wreath-like。只输出该英文翻译短语，不要解释，不要输出句子中其他内容，输出为一句话，首字母大写。"
            contrast_signal = call_llm(prompt_sys, snippet)
            if '.' not in contrast_signal:
                contrast_signal += '. '
            cache.append_log(PATHS['contrast_txt'], f"{patient}\n{contrast_signal}")
        
        hydro_signal = cache.hydro_cache.get(patient, '')
        if not hydro_signal:
            has_ventricle = '脑室' in finding and ('扩张' in finding or '扩大' in finding or '增宽' in finding)
            has_hydro = '积水' in impression
            
            if has_ventricle:
                snippet = "。".join(re.findall(r"[^，。,.]*脑室[^，。,.]*", finding))
                prompt_sys = "你是一位医学翻译专家。用户输入的句子可能包含多种内容，但都与脑室系统的状态有关，包括脑室扩张、受压或形态正常。请你仅提取并翻译与“脑室”相关的部分。如果同时存在形态正常和异常（如扩张或受压）的描述，只翻译异常部分（即扩张或受压）。如果同时包含多个异常（如扩张和受压），请分别将所有扩张部位合并到一句话中，使用句型：'Dilation of [多个结构] is observed.'；将所有受压部位合并到另一句话中，使用句型：'Compression of [多个结构] is observed.'。如果只提到脑室无异常，请统一翻译为：'No significant dilatation of the ventricular system.'。只输出英文句子，每句首字母需大写，句末加句号。不要添加解释，也不要输出原句中其他内容。
                hydro_signal = call_llm(prompt_sys, snippet)
                if '.' not in hydro_signal: hydro_signal += '.'
                
                cache.append_log(PATHS['hydro_txt'], f"{patient}\n{hydro_signal}")
                
            elif has_hydro:
                hydro_signal = 'Supratentorial hydrocephalus is noted.' if '幕上' in impression else 'Hydrocephalus is noted.'
                cache.append_log(PATHS['hydro_txt'], f"{patient}\n{hydro_signal}")

        location = cache.tumor_pos_cache.get(patient, '')
        if not location:
            prompt_sys = "提取文本中肿瘤位置,从下列选项中选择肿瘤位置属于的位置英文(可多选)。Parietal lobe, Frontal Lobe, Temporal Lobe, Insular Lobe, Limbic Lobe, Occipital Lobe, Subcortical Nuclei, Falx cerebri, corpus callosum, cerebellum, brainstem, Pineal region, Sellar region, suprasellar region, Third ventricle, spinal cord, Fourth ventricle, lateral ventricle, cranial fossa\n仅回答肿瘤左右以及位置英文(Right/Left xxx)。"
            location = call_llm(prompt_sys, impression) or 'brain'
            location = re.sub(r'[^a-zA-Z, ]', '', location).strip()
            cache.append_log(PATHS['tumor_pos_txt'], f"{patient}\n{location}")

        item['position'] = location

        shape = cache.shape_cache.get(patient, '')
        if not shape:
            prompt_sys = "你是一位医学翻译专家。用户输入的句子可能包含多种内容，请你提取其中描述**第一个病变位置的第一个 T1/T2 信号灶**形态的形容词，并用一个英文形容词表示。\n\n如果提到了病变形状，请从以下选项中选择最符合的一项：\n分叶状=lobulated，圆形=round，线状=linear，团块状=mass，结节状=nodular，斑片状=patchy，椭圆状=oval，花环状=wreath-like。\n如果没有提供病变形状，则只输出空字符串''。\n\n输出格式要求：\n- 只输出一个英文形容词，首字母小写。\n- 不要输出任何其他内容、解释或标点。"
            shape = call_llm(prompt_sys, finding)
            
            cache.append_log(PATHS['shape_txt'], f"{patient}\n{shape}")

        midline_signal = ''
        if '中线' in finding:
            snippet = re.search(r"[^，。,.]*中线[^，。,.]*", finding)
            snippet = snippet.group(0) if snippet else ''
            if '左' in snippet: midline_signal = 'Midline structures shift to the left. '
            elif '右' in snippet: midline_signal = 'Midline structures shift to the right. '
            else: midline_signal = 'No midline structure shift. '
            
        edema_signal = ''
        if '间质性' in impression and '水肿' in impression:
            edema_signal = 'Periventricular interstitial edema is noted. '

        template = f'In {location}'
        
        is_multiple = '多发' in impression
        if is_multiple:
            template += f", there are multiple {shape} lesions" if shape else ", there are multiple lesions"
            cnt_m += 1
        else:
            template += f", there is a {shape} lesion" if shape else ", there is a lesion"
            
        signals = []
        if t1_sig: signals.append(f"{t1_sig} in T1")
        if t2_sig: signals.append(f"{t2_sig} in T2")
        if fl_sig: signals.append(f"{fl_sig} in FLAIR")
        
        if signals:
            template += ' with ' + ', '.join(signals)
        template += '. '
        
        if contrast_signal: template += contrast_signal + ' '
        if hydro_signal: template += hydro_signal + ' '
        if edema_signal: template += edema_signal
        if midline_signal: template += midline_signal
        
        item['report'] = template.strip()
        
        results[patient] = item

    with open(PATHS['json_output'], 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

if __name__ == '__main__':
    main()