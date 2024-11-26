#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import kagglehub

# Download latest version
path = kagglehub.dataset_download("arunrk7/surface-crack-detection")
print("Path to dataset files:", path)

