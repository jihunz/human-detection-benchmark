import os
import xml.etree.ElementTree as ET

if __name__ == '__main__':
    result = []

    folder = '/Users/jihunjang/Downloads/ust/db/kisa/20190523-KISADB(배포판)/배포용_SA'
    for fname in os.listdir(folder):
        if not fname.endswith('.xml'):
            continue
        path = os.path.join(folder, fname)

        try:
            tree = ET.parse(path)
            root = tree.getroot()
            # 모든 AlarmDescription 탐색
            for desc in root.findall('.//AlarmDescription'):
                if desc.text and desc.text.strip() == "Falldown":
                    result.append(fname.replace('.xml', ''))
                    print(fname.replace('.xml', ''))
                    break  # 하나만 찾아도 해당 파일은 출력 후 다음 파일로 넘어감
        except Exception as e:
            print(f"[ERROR] {fname}: {e}")

    l = sorted(result)
    print(l)
