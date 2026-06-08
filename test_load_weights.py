import sys
sys.path.append('.')
try:
    from ai_wrapper import AdaFaceWrapper
    print("Loading AdaFaceWrapper...")
    model = AdaFaceWrapper(weight_path='weights/adaface_ir50_webface4m.ckpt', architecture='ir_50')
    print("Success!")
except Exception as e:
    import traceback
    traceback.print_exc()
