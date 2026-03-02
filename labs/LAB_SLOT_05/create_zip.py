"""
Script để tạo file ZIP nộp bài
Bao gồm: notebook, dataset đã chia, và dataset gốc
"""
import zipfile
import os
from pathlib import Path

def create_submission_zip():
    # Tên sinh viên
    student_name = "NguyenVanAnhDuy"
    
    # Đường dẫn thư mục hiện tại
    current_dir = Path(__file__).parent
    
    # Tên file ZIP
    zip_filename = f"{student_name}.zip"
    zip_path = current_dir / zip_filename
    
    # Danh sách các file cần nén
    files_to_zip = [
        f"{student_name}.ipynb",
        "gpa_study_hours.csv",  # Dataset gốc
        f"{student_name}_train.csv",  # Dataset train đã chia
        f"{student_name}_test.csv",  # Dataset test đã chia
    ]
    
    # Tạo file ZIP
    print("=" * 60)
    print(f"TẠO FILE ZIP: {zip_filename}")
    print("=" * 60)
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for filename in files_to_zip:
            file_path = current_dir / filename
            if file_path.exists():
                # Thêm file vào ZIP (chỉ lưu tên file, không lưu path)
                zipf.write(file_path, arcname=filename)
                print(f"✓ Đã thêm: {filename}")
            else:
                print(f"⚠ Không tìm thấy: {filename}")
    
    print("\n" + "=" * 60)
    print(f"✓ ĐÃ TẠO FILE ZIP THÀNH CÔNG!")
    print(f"Vị trí: {zip_path}")
    print(f"Kích thước: {zip_path.stat().st_size / 1024:.2f} KB")
    print("=" * 60)
    
    return zip_path

if __name__ == "__main__":
    create_submission_zip()
