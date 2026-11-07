from django.shortcuts import render
from .forms import UploadFileForm
from .utils import parse_voter_pdf
import os

# Create your views here.

def home_page(request):
    return render(request, 'index.html')




def upload_pdf_view(request):
  if request.method == 'POST':
    print('--- POST REQUEST AAYI HAI ---')
    form = UploadFileForm(request.POST, request.FILES)

    if form.is_valid():
      print('--- FORM VALID HAI ---')
      pdf_file = request.FILES['pdf_file']

      media_dir = 'media'
      os.makedirs(media_dir, exist_ok=True)
      file_path = os.path.join(media_dir, pdf_file.name)

      with open(file_path, 'wb+') as destination:
        for chunk in pdf_file.chunks():
          destination.write(chunk)

      print(f'File saved at: {file_path}. Ab parsing shuru ho rahi hai...')
      parse_voter_pdf(file_path)

      return render(
          request,
          'success.html',
          {'message': 'PDF data successfully extract ho gaya hai!'},
      )
    else:
      # Agar form galat hoga toh terminal par error dikhega
      print('--- FORM INVALID HAI ---')
      print(form.errors)
  else:
    form = UploadFileForm()

  return render(request, 'upload.html', {'form': form})
