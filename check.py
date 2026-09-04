import sys, torch, numpy, tokenizers, pytest, psutil

print('=== CODEXA ENVIRONMENT ===')
print('Python:', sys.version.split()[0])
print('Executable:', sys.executable)
print('PyTorch:', torch.__version__)
print('NumPy:', numpy.__version__)
print('Tokenizers:', tokenizers.__version__)
print('Pytest:', pytest.__version__)
print('RAM_GB:', round(psutil.virtual_memory().total/1024**3,2))
print('CUDA:', torch.cuda.is_available())
print('=== OK ===')