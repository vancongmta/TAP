import subprocess

def run_php_script(php_file, safe_path, unsafe_path, safe_token_file, unsafe_token_file):
    try:
        # Tạo lệnh để chạy tệp PHP với các tham số
        command = [
            'php', php_file,
            safe_path,  # Đường dẫn đến thư mục safe
            unsafe_path,  # Đường dẫn đến thư mục unsafe
            safe_token_file,  # Tên tệp token an toàn
            unsafe_token_file  # Tên tệp token không an toàn
        ]
        
        # Thực thi lệnh PHP
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        
        # In ra kết quả của lệnh PHP
        print("Kết quả (STDOUT):", result.stdout)
        print("Lỗi (STDERR):", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Đã xảy ra lỗi khi chạy tệp PHP: {e}")

# Thay đổi các tham số ở đây
php_file = 'Tokenizer.php'
safe_path = './text/'
unsafe_path = './unsafenew/'
safe_token_file = './safe.txt'
unsafe_token_file = './unsafenew.txt'

run_php_script(php_file, safe_path, unsafe_path, safe_token_file, unsafe_token_file)
