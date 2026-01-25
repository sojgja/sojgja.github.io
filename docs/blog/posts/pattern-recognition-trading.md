---
date: 2025-01-25
authors:
  - soigia
categories: [Technical Analysis, Machine Learning]
title: Pattern Recognition trong Trading - Phát hiện Chart Patterns với Python
description: >
  Sử dụng Python để nhận diện các chart patterns phổ biến: head and shoulders, triangles, flags và cách áp dụng trong trading.
---

# Pattern Recognition trong Trading

Chart patterns là công cụ mạnh mẽ trong technical analysis. Trong bài viết này, chúng ta sẽ implement pattern recognition với Python.

<!-- more -->

## Common Chart Patterns

1. **Head and Shoulders**
2. **Double Top/Bottom**
3. **Triangles** (Ascending, Descending, Symmetrical)
4. **Flags and Pennants**
5. **Cup and Handle**

## Implementation

```python
# patterns/pattern_recognition.py
import pandas as pd
import numpy as np
from scipy.signal import find_peaks

class PatternRecognizer:
    """Recognize chart patterns"""
    
    def find_head_shoulders(self, prices: pd.Series) -> List[Dict]:
        """Find head and shoulders pattern"""
        # Find peaks
        peaks, properties = find_peaks(prices, distance=10)
        
        patterns = []
        for i in range(len(peaks) - 2):
            left_shoulder = peaks[i]
            head = peaks[i + 1]
            right_shoulder = peaks[i + 2]
            
            # Check if head is higher than shoulders
            if (prices.iloc[head] > prices.iloc[left_shoulder] and
                prices.iloc[head] > prices.iloc[right_shoulder]):
                patterns.append({
                    'type': 'head_shoulders',
                    'left_shoulder': left_shoulder,
                    'head': head,
                    'right_shoulder': right_shoulder
                })
        
        return patterns
    
    def find_triangles(self, highs: pd.Series, lows: pd.Series) -> List[Dict]:
        """Find triangle patterns"""
        # Implementation
        pass
```

## Kết luận

Pattern recognition giúp identify trading opportunities!
