import zipfile

def unzip_data(src_path, dst_path):
  local_zip = src_path
  zip_ref = zipfile.ZipFile(local_zip, 'r')
  zip_ref.extractall(dst_path)
  zip_ref.close()