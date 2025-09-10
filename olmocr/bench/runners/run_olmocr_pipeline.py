import asyncio
import logging
from typing import Optional

# ใช้ openai client เพื่อสื่อสารกับ VLLM server
import openai
from olmocr.data.renderpdf import render_pdf_to_base64png

# ตั้งค่า logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("olmocr_vllm_runner")

# ตั้งค่า client ให้ชี้ไปที่ VLLM server ที่รันอยู่
# VLLM สร้าง server ที่เข้ากันได้กับ OpenAI API
client = openai.AsyncOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="vllm"  # ไม่จำเป็นต้องใช้ key จริง
)

# ชื่อโมเดลที่ VLLM กำลังให้บริการ
MODEL_NAME = "allenai/olmOCR-7B-0225-preview"
MAX_TOKENS = 4096 # กำหนด max tokens สำหรับ output

async def run_olmocr_pipeline(
    pdf_path: str, 
    page_num: int = 1, 
    model: Optional[str] = None, # รับ parameter model แต่จะใช้ค่าจาก VLLM server เป็นหลัก
    **kwargs # รับ arguments อื่นๆ ที่อาจส่งมาจาก convert.py
) -> Optional[str]:
    """
    ประมวลผล PDF หนึ่งหน้าโดยใช้ olmOCR ที่รันบน VLLM server

    Args:
        pdf_path: ที่อยู่ของไฟล์ PDF
        page_num: หมายเลขหน้า (เริ่มต้นที่ 1)
        model: ชื่อโมเดล (ไม่ถูกใช้งานโดยตรง แต่รับไว้เพื่อความเข้ากันได้)
        **kwargs: arguments อื่นๆ

    Returns:
        ข้อความ Markdown ที่ได้จากการประมวลผล หรือ None หากเกิดข้อผิดพลาด
    """
    logger.info(f"Processing {pdf_path}, page {page_num} with VLLM")
    try:
        # 1. แปลงหน้า PDF ที่ต้องการให้เป็นรูปภาพในรูปแบบ base64
        # ฟังก์ชันนี้มาจากไลบรารี olmocr เดิม
        base64_png = render_pdf_to_base64png(
            pdf_path, 
            page_num, 
            target_longest_image_dim=1024
        )

        # 2. สร้าง Prompt สำหรับโมเดล Multi-modal
        # เราจะส่งทั้งข้อความและรูปภาพไปพร้อมกัน
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Please transcribe this document page to markdown. Preserve all formatting, tables, and content."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_png}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=MAX_TOKENS,
            temperature=0.0,
        )

        # 3. ดึงผลลัพธ์ที่เป็น Markdown ออกมา
        markdown_output = response.choices[0].message.content
        logger.info(f"Successfully processed {pdf_path}, page {page_num}")
        return markdown_output

    except Exception as e:
        logger.error(f"Error processing {pdf_path}, page {page_num}: {type(e).__name__} - {str(e)}")
        # คืนค่า None หากเกิดข้อผิดพลาด เพื่อให้ convert.py สร้างไฟล์เปล่าตามตรรกะเดิม
        return None

# ส่วนนี้สำหรับการทดสอบไฟล์เดี่ยวๆ (ไม่จำเป็นสำหรับการทำงานร่วมกับ convert.py)
async def main():
    # ใส่ path ของ PDF ที่ต้องการทดสอบ
    pdf_path = "sample_data/pdfs/example.pdf"
    page_num = 1

    print(f"Testing with {pdf_path} page {page_num}")
    result = await run_olmocr_pipeline(pdf_path, page_num)
    
    if result:
        print("\n--- Extracted Markdown ---")
        print(result)
        print("--------------------------")
    else:
        print("Failed to extract text from the page.")

if __name__ == "__main__":
    # ตรวจสอบว่ามี VLLM server รันอยู่หรือไม่ก่อนทดสอบ
    try:
        asyncio.run(main())
    except openai.APIConnectionError:
        logger.error("Could not connect to VLLM server at http://localhost:8000/v1")
        logger.error("Please make sure the VLLM server is running using the command in the documentation.")