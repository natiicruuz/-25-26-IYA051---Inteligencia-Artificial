"""
Detección de tablas en documentos.

Este módulo detecta tablas en imágenes de documentos usando operaciones
morfológicas para encontrar líneas horizontales y verticales.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import os


class TableDetector:
    """
    Detector de tablas usando operaciones morfológicas.
    
    Detecta estructuras rectangulares con líneas horizontales y verticales
    que puedan corresponder a tablas.
    """
    
    def __init__(
        self,
        line_min_length: int = 100,
        line_thickness: int = 1,
        cell_min_area: int = 100
    ):
        """
        Inicializa el detector de tablas.
        
        Args:
            line_min_length: Longitud mínima para considerar una línea válida.
            line_thickness: Grosor para detectar líneas.
            cell_min_area: Área mínima para considerar una celda válida.
        """
        self.line_min_length = line_min_length
        self.line_thickness = line_thickness
        self.cell_min_area = cell_min_area
    
    def detect_tables(
        self,
        image_path: str
    ) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """
        Detecta tablas en una imagen.
        
        Args:
            image_path: Ruta a la imagen del documento.
        
        Returns:
            Lista de tuplas (imagen_tabla, bounding_box).
        """
        # Cargar imagen
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"No se pudo cargar la imagen: {image_path}")
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Binarización
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Detectar líneas horizontales y verticales
        horizontal_lines = self._detect_horizontal_lines(binary)
        vertical_lines = self._detect_vertical_lines(binary)
        
        # Combinar líneas para encontrar estructuras de tabla
        table_mask = cv2.add(horizontal_lines, vertical_lines)
        
        # Encontrar contornos de tablas potenciales
        contours, _ = cv2.findContours(
            table_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        tables = []
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filtrar por tamaño (debe ser suficientemente grande para ser tabla)
            if w * h > 10000:  # Área mínima de tabla
                table_img = image[y:y+h, x:x+w]
                tables.append((table_img, (x, y, w, h)))
        
        return tables
    
    def _detect_horizontal_lines(self, binary: np.ndarray) -> np.ndarray:
        """
        Detecta líneas horizontales en la imagen.
        
        Args:
            binary: Imagen binarizada.
        
        Returns:
            Máscara con líneas horizontales.
        """
        # Kernel horizontal para detectar líneas horizontales
        horizontal_size = binary.shape[1] // 30
        horizontal_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (horizontal_size, self.line_thickness)
        )
        
        # Detectar líneas horizontales
        horizontal_lines = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            horizontal_kernel,
            iterations=2
        )
        
        return horizontal_lines
    
    def _detect_vertical_lines(self, binary: np.ndarray) -> np.ndarray:
        """
        Detecta líneas verticales en la imagen.
        
        Args:
            binary: Imagen binarizada.
        
        Returns:
            Máscara con líneas verticales.
        """
        # Kernel vertical para detectar líneas verticales
        vertical_size = binary.shape[0] // 30
        vertical_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (self.line_thickness, vertical_size)
        )
        
        # Detectar líneas verticales
        vertical_lines = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            vertical_kernel,
            iterations=2
        )
        
        return vertical_lines
    
    def export_to_markdown(
        self,
        num_rows: int,
        num_cols: int,
        output_path: str
    ):
        """
        Exporta una estructura de tabla simple a formato Markdown.
        
        Args:
            num_rows: Número de filas detectadas.
            num_cols: Número de columnas detectadas.
            output_path: Ruta donde guardar el archivo Markdown.
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            # Header
            header = "| " + " | ".join([f"Col {i+1}" for i in range(num_cols)]) + " |"
            f.write(header + "\n")
            
            # Separator
            separator = "|" + "|".join(["---" for _ in range(num_cols)]) + "|"
            f.write(separator + "\n")
            
            # Rows
            for row in range(num_rows):
                row_str = "| " + " | ".join([f"Celda {row+1},{col+1}" for col in range(num_cols)]) + " |"
                f.write(row_str + "\n")
    
    def visualize_detections(
        self,
        image_path: str,
        save_path: Optional[str] = None
    ):
        """
        Visualiza las tablas detectadas en la imagen.
        
        Args:
            image_path: Ruta a la imagen del documento.
            save_path: Ruta opcional donde guardar la visualización.
        """
        import matplotlib.pyplot as plt
        
        # Cargar imagen
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detectar tablas
        tables = self.detect_tables(image_path)
        
        # Dibujar bounding boxes
        for idx, (_, (x, y, w, h)) in enumerate(tables):
            cv2.rectangle(image_rgb, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.putText(
                image_rgb,
                f"Tabla {idx+1}",
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2
            )
        
        # Visualizar
        plt.figure(figsize=(15, 10))
        plt.imshow(image_rgb)
        plt.title(f'Tablas Detectadas: {len(tables)}', fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return tables
    
    def save_tables(
        self,
        image_path: str,
        output_dir: str = './extracted_tables'
    ) -> List[str]:
        """
        Detecta y guarda las tablas como imágenes separadas.
        
        Args:
            image_path: Ruta a la imagen del documento.
            output_dir: Directorio donde guardar las tablas extraídas.
        
        Returns:
            Lista de rutas a las tablas guardadas.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Detectar tablas
        tables = self.detect_tables(image_path)
        
        saved_paths = []
        
        for idx, (table_img, bbox) in enumerate(tables):
            output_path = os.path.join(output_dir, f'table_{idx+1}.png')
            cv2.imwrite(output_path, table_img)
            saved_paths.append(output_path)
            
            # También crear estructura Markdown básica
            markdown_path = os.path.join(output_dir, f'table_{idx+1}.md')
            # Estimación simple: 3 filas, 3 columnas
            self.export_to_markdown(3, 3, markdown_path)
        
        print(f"\n{'='*60}")
        print(f"Tablas Extraídas: {len(tables)}")
        print(f"{'='*60}")
        for i, path in enumerate(saved_paths):
            print(f"  Tabla {i+1}: {path}")
        print(f"{'='*60}\n")
        
        return saved_paths


def main():
    """Función de ejemplo para usar el detector de tablas."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Detectar tablas en un documento'
    )
    
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Ruta a la imagen del documento'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./extracted_tables',
        help='Directorio donde guardar las tablas (default: ./extracted_tables)'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Mostrar visualización de las tablas detectadas'
    )
    
    args = parser.parse_args()
    
    # Crear detector
    detector = TableDetector()
    
    # Visualizar si se solicita
    if args.visualize:
        detector.visualize_detections(args.image)
    
    # Guardar tablas
    saved_paths = detector.save_tables(args.image, args.output_dir)
    
    return saved_paths


if __name__ == '__main__':
    main()