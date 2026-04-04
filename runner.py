from supabase import create_client
import os

supabase_url = "https://ecypnsmehlznwbfongxt.supabase.co"
supabase_key = "<sb_secret_xp8VMw91Nw_utNTE0OGCQQ_fjmhQejs>"  # استخدم SERVICE KEY للPod
supabase = create_client(supabase_url, supabase_key)

os.makedirs("input/images", exist_ok=True)

# جلب كل الملفات من bucket
bucket_files = supabase.storage.from_('images').list()
for f in bucket_files:
    file_path = f"input/images/{f['name']}"
    supabase.storage.from_('images').download(f['name'], file_path)

print("All images downloaded from Supabase!")
