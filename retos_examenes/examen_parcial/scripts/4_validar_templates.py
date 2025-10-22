import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vision.template_matching import get_template_library

library = get_template_library()

if library.is_loaded():
    print("✅ Templates cargados correctamente")
    print(f"   Valores: {len(library.value_templates)}/13")
    print(f"   Palos: {len(library.suit_templates)}/4")
else:
    print("❌ Error al cargar templates")