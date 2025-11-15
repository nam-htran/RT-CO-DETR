import os

def save_all_source_code(output_path='utils/output.txt', excluded_dirs=None, excluded_files=None, included_ext=None):
    if excluded_dirs is None:
        excluded_dirs = {
            '__pycache__', '_dataset', '_logs', '_model_trained', '_results',
            'node_modules', '.next', '.git', '.venv', '.vscode', '.idea', 'dist', 'build', 'taco_yolo', 'data_input', 'wandb', 'LICENSE',
        }
    if excluded_files is None:
        excluded_files = {
            os.path.basename(output_path),
            '.gitignore', 'pyproject.toml', 'credentials.py', 'read_file_for_gpt.py', '.env',
            'package-lock.json'
        }
    if included_ext is None:
        included_ext = {'.py', '.js', '.ts', '.jsx', '.tsx', '.json', '.html', '.css'}

    file_count = 0
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for root, dirs, files in os.walk('.'):
            # Bỏ thư mục không cần thiết
            dirs[:] = [d for d in dirs if d not in excluded_dirs]

            for file in files:
                if any(file.endswith(ext) for ext in included_ext) and file not in excluded_files:
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            code = f.read()
                        outfile.write(f"\n===== {filepath} =====\n")
                        outfile.write(code + "\n")
                        file_count += 1
                    except Exception as e:
                        print(f"⚠️ Lỗi đọc {filepath}: {e}")

        # Ghi phần cây thư mục
        outfile.write("\n===== DIRECTORY TREE =====\n")
        for dirpath, dirnames, filenames in os.walk('.'):
            dirnames[:] = [d for d in dirnames if d not in excluded_dirs]
            level = dirpath.count(os.sep)
            indent = '    ' * level
            outfile.write(f"{indent}{os.path.basename(dirpath)}/\n")
            for file in filenames:
                if file not in excluded_files:
                    subindent = '    ' * (level + 1)
                    outfile.write(f"{subindent}{file}\n")

    print(f"✅ Đã lưu {file_count} file mã nguồn (backend + frontend) vào '{output_path}'.")

if __name__ == '__main__':
    save_all_source_code()
