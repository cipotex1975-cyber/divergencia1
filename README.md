# Multi-Timeframe Divergence Trading Strategy

Este proyecto implementa una estrategia de trading algorítmico multi-timeframe utilizando Backtrader y Python.

## 📋 Descripción

La estrategia combina análisis técnico en tres marcos temporales diferentes:

- **4 Horas (4H)**: Identificación de niveles de soporte y resistencia
- **1 Hora (1H)**: Detección de divergencias MACD cerca de niveles clave
- **15 Minutos (15M)**: Ejecución de entradas en rupturas confirmadas

## 🎯 Lógica de la Estrategia

### Paso 1: Análisis de Contexto (4H)
- Detecta niveles de soporte y resistencia usando swing highs/lows
- Identifica zonas clave donde el precio podría reaccionar

### Paso 2: Señal de Preparación (4H + 1H)
- Verifica si el precio está cerca de un nivel S/R (dentro del 2%)
- Detecta divergencias MACD en el timeframe de 1H:
  - **Divergencia Alcista**: Precio hace mínimos más bajos, MACD hace mínimos más altos (cerca de soporte)
  - **Divergencia Bajista**: Precio hace máximos más altos, MACD hace máximos más bajos (cerca de resistencia)

### Paso 3: Identificación de Nivel de Entrada
- **Setup Alcista**: Identifica el último máximo relevante a romper
- **Setup Bajista**: Identifica el último mínimo relevante a romper

### Paso 4: Ejecución (15M)
- Espera la ruptura del nivel identificado en el timeframe de 15 minutos
- Ejecuta la entrada con gestión de riesgo:
  - Stop Loss: 2%
  - Take Profit: 6% (Ratio 3:1)

## 🚀 Instalación

1. Clonar o descargar el proyecto

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## 📁 Estructura del Proyecto

```
desarrollo_divergencias/
├── config.py              # Configuración de parámetros
├── data_loader.py         # Descarga de datos con yfinance
├── indicators.py          # Indicadores técnicos y detección de divergencias
├── strategy.py            # Estrategia principal de Backtrader
├── main.py                # Script de ejecución principal
├── tickers.txt            # Lista de activos a analizar
├── requirements.txt       # Dependencias del proyecto
└── results/               # Directorio para resultados (se crea automáticamente)
```

## ⚙️ Configuración

Edita `config.py` para ajustar los parámetros de la estrategia:

```python
# Parámetros de Soporte/Resistencia
SR_LOOKBACK_PERIOD = 20
SR_PROXIMITY_PERCENT = 2.0

# Parámetros MACD
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Gestión de Riesgo
STOP_LOSS_PERCENT = 2.0
TAKE_PROFIT_PERCENT = 6.0
POSITION_SIZE_PERCENT = 10.0
```

## 📊 Uso

### 1. Configurar Activos

Edita `tickers.txt` con los símbolos que deseas analizar (uno por línea):
```
AAPL
MSFT
GOOGL
TSLA
```

### 2. Ejecutar Backtest

```bash
python main.py
```

### 3. Resultados

El script generará:
- Análisis detallado por cada ticker en consola
- Archivo CSV con resumen de todos los backtests
- Archivo CSV con todas las señales detectadas
- Gráficas de los resultados (si está habilitado)

Los resultados se guardan en el directorio `results/`:
- `backtest_summary_YYYYMMDD_HHMMSS.csv`: Resumen de rendimiento
- `signals_YYYYMMDD_HHMMSS.csv`: Señales detectadas

## 📈 Métricas de Rendimiento

El backtest proporciona las siguientes métricas:

- **Valor Final del Portfolio**: Capital final después del backtest
- **Sharpe Ratio**: Rendimiento ajustado por riesgo
- **Max Drawdown**: Máxima caída desde un pico
- **Retorno Total**: Rendimiento total del período
- **Total de Operaciones**: Número de trades ejecutados
- **Win Rate**: Porcentaje de operaciones ganadoras

## 🔧 Personalización

### Añadir Nuevos Indicadores

Edita `indicators.py` para añadir nuevos indicadores técnicos:

```python
class CustomIndicator:
    def calculate(self, data):
        # Tu lógica aquí
        pass
```

### Modificar Lógica de Entrada

Edita `strategy.py`, específicamente los métodos:
- `_detect_bullish_divergence()`: Lógica de divergencia alcista
- `_detect_bearish_divergence()`: Lógica de divergencia bajista
- `_check_entry_breakout()`: Condiciones de entrada

### Cambiar Timeframes

Modifica `config.py` y `data_loader.py` para usar diferentes intervalos.

## 📝 Notas Importantes

1. **Datos Históricos**: yfinance tiene limitaciones en la cantidad de datos históricos para intervalos pequeños (15M, 1H)
2. **Comisiones**: El backtest incluye comisiones del 0.1% por operación
3. **Slippage**: No se incluye slippage en esta versión básica
4. **Short Selling**: La versión actual solo implementa operaciones long (compra)

## 🐛 Solución de Problemas

### Error al descargar datos
- Verifica tu conexión a internet
- Algunos tickers pueden no tener datos disponibles en todos los timeframes
- yfinance puede tener límites de tasa de descarga

### Sin señales detectadas
- Ajusta los parámetros de sensibilidad en `config.py`
- Verifica que haya suficientes datos históricos
- Revisa los umbrales de proximidad a S/R

## 📚 Recursos Adicionales

- [Documentación de Backtrader](https://www.backtrader.com/docu/)
- [yfinance Documentation](https://pypi.org/project/yfinance/)
- [MACD Divergence Trading](https://www.investopedia.com/terms/d/divergence.asp)

## ⚠️ Disclaimer

Este código es solo para fines educativos y de investigación. No constituye asesoramiento financiero. El trading conlleva riesgos significativos y puede resultar en pérdidas. Siempre realiza tu propia investigación y consulta con profesionales financieros antes de operar con dinero real.

## 📄 Licencia

Este proyecto es de código abierto y está disponible bajo la licencia MIT.
"# divergencias" 
