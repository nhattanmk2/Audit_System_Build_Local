import streamlit as st
import os
import zipfile
import io
from docx_utils import split_reports
from pathlib import Path

# Paths
BASE_DIR = Path("e:/CIS_Audit_Project")
TEMPLATE_DIR = BASE_DIR / "CIS_Data" / "Reports_Template"
TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="Report Splitter", layout="centered")
st.title("📄 Báo cáo Kiểm định - Công cụ Tách File")

st.markdown("""
Công cụ này dùng để tách một file báo cáo tổng hợp chứa nhiều hệ thống (Server, Switch, Firewall) thành các file báo cáo riêng lẻ cho từng hệ thống.
Các file sau khi tách sẽ được lưu vào thư mục `CIS_Data/Reports_Template/` để dùng cho tính năng điền kết quả tự động.
""")

uploaded_file = st.file_uploader("Upload file Báo cáo mẫu (.docx)", type=["docx"])

if uploaded_file:
    # Save the uploaded file temporarily
    temp_template_path = BASE_DIR / "CIS_Data" / "temp_master_template.docx"
    with open(temp_template_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.button("🚀 Thực hiện Tách File", type="primary"):
        with st.spinner("Đang xử lý tách báo cáo..."):
            try:
                # Call splitting function
                generated_files = split_reports(str(temp_template_path), str(TEMPLATE_DIR))
                
                num_files = len(generated_files)
                st.success(f"Đã tách thành công {num_files} báo cáo hệ thống!")
                
                # Show list of generated files
                st.subheader("Danh sách hệ thống đã nhận diện:")
                for f in generated_files:
                    st.write(f"- {Path(f).name}")
                
                # Create a ZIP for download
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w") as z:
                    for f in generated_files:
                        z.write(f, Path(f).name)
                
                st.download_button(
                    label="📥 Tải về file ZIP (30 hệ thống)",
                    data=zip_buffer.getvalue(),
                    file_name="Cmc_Reports_Templates.zip",
                    mime="application/zip"
                )
                
                st.info(f"Các file đã được lưu tĩnh tại: `{TEMPLATE_DIR.relative_to(BASE_DIR)}`")

            except Exception as e:
                st.error(f"Có lỗi xảy ra: {e}")
                import traceback
                st.code(traceback.format_exc())

st.divider()
st.markdown("© 2026 CMC Cyber Security")
