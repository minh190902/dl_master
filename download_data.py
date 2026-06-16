"""Tải dataset UIT-VSFC (Vietnamese Students' Feedback Corpus) về data/UIT-VSFC/.

Nguồn: HuggingFace `uitnlp/vietnamese_students_feedback` (các file gốc lưu trên Google Drive).
Chạy:  python download_data.py
"""
import os
import gdown

BASE = os.path.join(os.path.dirname(__file__), "data", "UIT-VSFC")

# Google Drive file IDs cho từng split / loại file (sents, sentiments, topics).
FILES = {
    "train": {
        "sents":      "1nzak5OkrheRV1ltOGCXkT671bmjODLhP",
        "sentiments": "1ye-gOZIBqXdKOoi_YxvpT6FeRNmViPPv",
        "topics":     "14MuDtwMnNOcr4z_8KdpxprjbwaQ7lJ_C",
    },
    "dev": {
        "sents":      "1sMJSR3oRfPc3fe1gK-V3W5F24tov_517",
        "sentiments": "1GiY1AOp41dLXIIkgES4422AuDwmbUseL",
        "topics":     "1DwLgDEaFWQe8mOd7EpF-xqMEbDLfdT-W",
    },
    "test": {
        "sents":      "1aNMOeZZbNwSRkjyCWAGtNCMa3YrshR-n",
        "sentiments": "1vkQS5gI0is4ACU58-AbWusnemw7KZNfO",
        "topics":     "1_ArMpDguVsbUGl-xSMkTF_p5KpZrmpSB",
    },
}


def main():
    for split, files in FILES.items():
        out_dir = os.path.join(BASE, split)
        os.makedirs(out_dir, exist_ok=True)
        for kind, file_id in files.items():
            out_path = os.path.join(out_dir, f"{kind}.txt")
            if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                print(f"[skip] {out_path} đã tồn tại")
                continue
            url = f"https://drive.google.com/uc?id={file_id}"
            print(f"[get ] {split}/{kind} -> {out_path}")
            gdown.download(url, out_path, quiet=True)

    print("\n=== Kiểm tra file đã tải ===")
    for split in FILES:
        for kind in ("sents", "sentiments", "topics"):
            p = os.path.join(BASE, split, f"{kind}.txt")
            n = sum(1 for _ in open(p, encoding="utf-8")) if os.path.exists(p) else -1
            print(f"{split:5s} {kind:10s}: {n:6d} dòng")


if __name__ == "__main__":
    main()
