"""
Script de verificación para dataset normalizado.
"""

import sys
import argparse

import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.normalized_dataset import verify_normalized_dataset


def main():
    parser = argparse.ArgumentParser(
        description='Verificar dataset normalizado'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Ruta al directorio del dataset normalizado'
    )
    
    args = parser.parse_args()
    
    success = verify_normalized_dataset(args.dataset)
    
    if success:
        print("\n✅ Dataset listo para entrenar")
        print("\nComando para entrenar:")
        print(f'python training\\train_normalized.py --dataset "{args.dataset}" --epochs 50 --patience 10\n')
    else:
        print("\n❌ Hay problemas con el dataset")
        sys.exit(1)


if __name__ == '__main__':
    main()