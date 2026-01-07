"""
Arquitectura de red neuronal convolucional para OCR.

Este módulo define el modelo CNN para clasificación de caracteres manuscritos.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List
from datetime import datetime
from models.config import OCRConfig


class OCRCNN(nn.Module):
    """
    Red Neuronal Convolucional para reconocimiento de caracteres.
    
    Arquitectura:
        - 3 bloques convolucionales (32, 64, 96 filtros)
        - Batch Normalization y Dropout
        - 3 capas fully connected (320, 160, num_classes)
    """
    
    def __init__(self, config: OCRConfig):
        """
        Inicializa la arquitectura del modelo.
        
        Args:
            config: Configuración del modelo con input_shape y num_classes.
        """
        super(OCRCNN, self).__init__()
        
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Construir capas convolucionales
        self.conv_layers = self._build_conv_layers()
        
        # Calcular tamaño de salida de las capas convolucionales
        # Después de 3 MaxPool(2x2): 28 -> 14 -> 7 -> 3
        self.conv_output_size = 96 * 3 * 3  # 864
        
        # Construir capas fully connected
        self.fc_layers = self._build_fc_layers()
        
        # Función de pérdida y optimizador
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=config.learning_rate)
        
        # Mover modelo al dispositivo
        self.to(self.device)
    
    def _build_conv_layers(self) -> nn.Sequential:
        """
        Construye las capas convolucionales del modelo.
        
        Returns:
            Sequential con los bloques convolucionales.
        """
        return nn.Sequential(
            # Bloque convolucional 1: 1 -> 32 filtros
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2),  # 28x28 -> 14x14
            nn.Dropout2d(0.3),
            
            # Bloque convolucional 2: 32 -> 64 filtros
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),  # 14x14 -> 7x7
            nn.Dropout2d(0.3),
            
            # Bloque convolucional 3: 64 -> 96 filtros
            nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(96),
            nn.MaxPool2d(kernel_size=2),  # 7x7 -> 3x3
            nn.Dropout2d(0.3),
        )
    
    def _build_fc_layers(self) -> nn.Sequential:
        """
        Construye las capas fully connected del modelo.
        
        Returns:
            Sequential con las capas densas.
        """
        return nn.Sequential(
            nn.Flatten(),
            
            # Primera capa densa: 864 -> 320
            nn.Linear(self.conv_output_size, 320),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            # Segunda capa densa: 320 -> 160
            nn.Linear(320, 160),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            # Capa de salida: 160 -> num_classes
            nn.Linear(160, self.config.num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass del modelo.
        
        Args:
            x: Tensor de entrada (batch_size, 1, 28, 28).
        
        Returns:
            Tensor de salida (batch_size, num_classes).
        """
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
    
    def train_model(
        self,
        train_loader,
        val_loader: Optional = None,
        epochs: int = 10,
        verbose: bool = True,
        early_stopping: bool = True,
        patience: int = 5,
        checkpoint_dir: str = './checkpoints'
    ) -> Tuple[List[float], List[float], dict]:
        """
        Entrena el modelo con early stopping opcional.
        
        Args:
            train_loader: DataLoader de entrenamiento.
            val_loader: DataLoader de validación opcional.
            epochs: Número máximo de épocas.
            verbose: Si True, muestra progreso detallado.
            early_stopping: Si True, activa early stopping.
            patience: Épocas a esperar sin mejora antes de parar.
            checkpoint_dir: Directorio donde guardar checkpoints.
        
        Returns:
            Tupla con (train_losses, val_losses, early_stop_info).
        """
        import os
        
        # Crear directorio de checkpoints si no existe
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Inicializar early stopping si está activado
        early_stopper = None
        best_model_path = None
        
        if early_stopping and val_loader:
            early_stopper = EarlyStopping(patience=patience, verbose=verbose)
            # Ruta donde guardar el mejor modelo
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            best_model_path = os.path.join(checkpoint_dir, f'best_model_{timestamp}.pth')
            if verbose:
                print(f"\n📊 Early Stopping activado:")
                print(f"  → Patience: {patience} épocas")
                print(f"  → Checkpoint: {best_model_path}\n")
        
        self.train()
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # ==================== ENTRENAMIENTO ====================
            running_loss = 0.0
            correct = 0
            total = 0
            
            # Barra de progreso
            if verbose:
                loader = tqdm(
                    train_loader,
                    desc=f"Época {epoch + 1}/{epochs}",
                    unit="batch"
                )
            else:
                loader = train_loader
            
            for images, labels in loader:
                # Mover datos al dispositivo
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self(images)
                loss = self.criterion(outputs, labels)
                
                # Backward pass y optimización
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Estadísticas
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                if verbose and isinstance(loader, tqdm):
                    loader.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'acc': f'{100 * correct / total:.2f}%'
                    })
            
            # Pérdida y precisión promedio de la época
            epoch_loss = running_loss / len(train_loader)
            epoch_acc = 100 * correct / total
            train_losses.append(epoch_loss)
            
            # ==================== VALIDACIÓN ====================
            val_loss = None
            if val_loader:
                val_loss = self.evaluate_model(val_loader, verbose=False)
                val_losses.append(val_loss)
                
                if verbose:
                    print(f"\n📊 Época [{epoch + 1}/{epochs}]:")
                    print(f"  Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}%")
                    print(f"  Val Loss:   {val_loss:.4f}")
            else:
                if verbose:
                    print(f"\n📊 Época [{epoch + 1}/{epochs}]:")
                    print(f"  Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")
            
            # ==================== EARLY STOPPING ====================
            if early_stopper and val_loss is not None:
                # Verificar si es el mejor modelo hasta ahora
                if val_loss == early_stopper.best_loss or early_stopper.best_loss is None:
                    # Guardar checkpoint del mejor modelo
                    self.save_checkpoint(best_model_path)
                    if verbose:
                        print(f"  💾 Mejor modelo guardado")
                
                # Evaluar early stopping
                should_stop = early_stopper(val_loss, epoch + 1)
                
                if should_stop:
                    if verbose:
                        print(f"\n{'='*60}")
                        print(f"⏹️  Entrenamiento detenido por Early Stopping")
                        print(f"{'='*60}")
                    break
        
        # ==================== RESTAURAR MEJOR MODELO ====================
        if early_stopper and best_model_path and os.path.exists(best_model_path):
            if verbose:
                print(f"\n🔄 Restaurando mejor modelo desde época {early_stopper.best_epoch}...")
            self.load_checkpoint(best_model_path)
            if verbose:
                print(f"✓ Modelo restaurado correctamente")
        
        # Información de early stopping
        early_stop_info = {}
        if early_stopper:
            early_stop_info = early_stopper.get_info()
            early_stop_info['checkpoint_path'] = best_model_path
        
        return train_losses, val_losses, early_stop_info
    
    def evaluate_model(self, test_loader, verbose: bool = True) -> float: 
        """
        Evalúa el modelo en un conjunto de datos.
        
        Args:
            test_loader: DataLoader de test.
            verbose: Si True, imprime resultados.
        
        Returns:
            Pérdida promedio en el conjunto de test.
        """
        self.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self(images)
                loss = self.criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = test_loss / len(test_loader)
        accuracy = 100 * correct / total
        
        if verbose:
            print(f'\nEvaluación en test set:')
            print(f'  Pérdida: {avg_loss:.4f}')
            print(f'  Precisión: {accuracy:.2f}%')
        
        return avg_loss
    
    def save_checkpoint(self, path: str):
        """
        Guarda el modelo entrenado.
        
        Args:
            path: Ruta donde guardar el checkpoint.
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, path)
        print(f"Modelo guardado en: {path}")
    
    def load_checkpoint(self, path: str):
        """
        Carga un modelo previamente guardado.
        
        Args:
            path: Ruta del checkpoint a cargar.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Modelo cargado desde: {path}")


class EarlyStopping:
    """
    Monitorea la pérdida de validación y para el entrenamiento cuando
    deja de mejorar después de 'patience' épocas consecutivas.
    """
    
    def __init__(self, patience: int = 5, min_delta: float = 0.0, verbose: bool = True):
        """
        Args:
            patience: Número de épocas a esperar sin mejora antes de parar.
            min_delta: Cambio mínimo en val_loss para considerar mejora.
            verbose: Si True, imprime mensajes informativos.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        
        self.counter = 0  # Contador de épocas sin mejora
        self.best_loss = None  # Mejor pérdida registrada
        self.early_stop = False  # Flag para indicar si debe parar
        self.best_epoch = 0  # Época con mejor pérdida
    
    def __call__(self, val_loss: float, epoch: int) -> bool:
        """
        Evalúa si el entrenamiento debe continuar o parar.
        
        Args:
            val_loss: Pérdida de validación actual.
            epoch: Número de época actual.
        
        Returns:
            True si debe parar el entrenamiento, False si debe continuar.
        """
        # Primera época: inicializar
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_epoch = epoch
            if self.verbose:
                print(f"  → Early Stopping inicializado (baseline: {val_loss:.4f})")
            return False
        
        # Verificar si hay mejora (considerando min_delta)
        if val_loss < (self.best_loss - self.min_delta):
            # ¡Hay mejora!
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
            if self.verbose:
                print(f"  ✓ Val loss mejoró a {val_loss:.4f} (mejor hasta ahora)")
            return False
        else:
            # No hay mejora
            self.counter += 1
            if self.verbose:
                print(f"  ⚠️  Val loss no mejoró ({self.counter}/{self.patience})")
            
            # ¿Se alcanzó el límite de patience?
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"\n  🛑 EARLY STOPPING activado")
                    print(f"  → Mejor val_loss: {self.best_loss:.4f} (época {self.best_epoch})")
                return True
            
            return False
    
    def get_info(self) -> dict:
        """Retorna información del estado del early stopping."""
        return {
            'best_loss': self.best_loss,
            'best_epoch': self.best_epoch,
            'counter': self.counter,
            'stopped': self.early_stop
        }