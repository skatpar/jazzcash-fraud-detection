#!/usr/bin/env python3
"""
Model Component Management Utilities

This module provides utilities for saving and loading all preprocessing components
and trained models for the fraud detection system.

Author: AI Assistant
Date: November 2025
"""

import json
import os
import logging
from datetime import datetime
from typing import Dict, Any, List

from pyspark.sql import SparkSession, DataFrame
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.feature import VectorAssembler, StandardScalerModel, StringIndexer, OneHotEncoder


class ModelComponentManager:
    """
    Manager class for saving and loading all model components
    """
    
    def __init__(self, base_path: str, model_name: str = "fraud_model"):
        """
        Initialize the model component manager
        
        Args:
            base_path: Base directory for saving components
            model_name: Name prefix for the model and components
        """
        self.base_path = base_path
        self.model_name = model_name
        self.model_path = f"{base_path}/{model_name}"
        self.preprocessing_path = f"{base_path}/preprocessing_{model_name}"
        
        # Setup logging
        self.logger = logging.getLogger('ModelComponentManager')
        
    def save_all_components(self, 
                          model: LogisticRegressionModel,
                          categorical_pipeline: Pipeline,
                          vector_assembler: VectorAssembler,
                          scaler_model: StandardScalerModel,
                          feature_metadata: Dict[str, Any],
                          training_df: DataFrame = None) -> Dict[str, str]:
        """
        Save all model components and preprocessing artifacts
        
        Args:
            model: Trained LogisticRegressionModel
            categorical_pipeline: Fitted categorical preprocessing pipeline
            vector_assembler: VectorAssembler for feature assembly
            scaler_model: Fitted StandardScalerModel
            feature_metadata: Dictionary containing feature information
            training_df: Original training DataFrame (optional, for validation)
        
        Returns:
            Dict containing paths to all saved components
        """
        self.logger.info("💾 Starting to save all model components...")
        
        # Create directories
        os.makedirs(self.preprocessing_path, exist_ok=True)
        
        saved_paths = {}
        
        try:
            # 1. Save the trained model
            model.write().overwrite().save(self.model_path)
            saved_paths['model'] = self.model_path
            self.logger.info(f"✅ Model saved to {self.model_path}")
            
            # 2. Save categorical preprocessing pipeline
            cat_pipeline_path = f"{self.preprocessing_path}/categorical_pipeline"
            categorical_pipeline.write().overwrite().save(cat_pipeline_path)
            saved_paths['categorical_pipeline'] = cat_pipeline_path
            self.logger.info(f"✅ Categorical pipeline saved to {cat_pipeline_path}")
            
            # 3. Save vector assembler
            assembler_path = f"{self.preprocessing_path}/vector_assembler"
            vector_assembler.write().overwrite().save(assembler_path)
            saved_paths['vector_assembler'] = assembler_path
            self.logger.info(f"✅ Vector assembler saved to {assembler_path}")
            
            # 4. Save standard scaler
            scaler_path = f"{self.preprocessing_path}/standard_scaler"
            scaler_model.write().overwrite().save(scaler_path)
            saved_paths['standard_scaler'] = scaler_path
            self.logger.info(f"✅ Standard scaler saved to {scaler_path}")
            
            # 5. Save feature metadata
            enhanced_metadata = {
                **feature_metadata,
                "save_timestamp": datetime.now().isoformat(),
                "model_name": self.model_name,
                "component_paths": saved_paths
            }
            
            metadata_path = f"{self.preprocessing_path}/feature_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(enhanced_metadata, f, indent=2)
            saved_paths['metadata'] = metadata_path
            self.logger.info(f"✅ Feature metadata saved to {metadata_path}")
            
            # 6. Save component manifest
            manifest = {
                "model_name": self.model_name,
                "save_date": datetime.now().isoformat(),
                "components": list(saved_paths.keys()),
                "paths": saved_paths,
                "total_components": len(saved_paths)
            }
            
            manifest_path = f"{self.preprocessing_path}/component_manifest.json"
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            saved_paths['manifest'] = manifest_path
            self.logger.info(f"✅ Component manifest saved to {manifest_path}")
            
            self.logger.info("🎯 All components saved successfully!")
            self.logger.info(f"📦 Total components saved: {len(saved_paths)}")
            
            return saved_paths
            
        except Exception as e:
            self.logger.error(f"❌ Error saving components: {str(e)}")
            raise
    
    def load_all_components(self) -> Dict[str, Any]:
        """
        Load all saved components for inference
        
        Returns:
            Dictionary containing all loaded components
        """
        self.logger.info("📦 Loading all model components...")
        
        components = {}
        
        try:
            # Load component manifest first
            manifest_path = f"{self.preprocessing_path}/component_manifest.json"
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            components['manifest'] = manifest
            self.logger.info(f"✅ Component manifest loaded")
            
            # Load model
            components['model'] = LogisticRegressionModel.load(self.model_path)
            self.logger.info(f"✅ Model loaded from {self.model_path}")
            
            # Load categorical pipeline
            cat_pipeline_path = f"{self.preprocessing_path}/categorical_pipeline"
            components['categorical_pipeline'] = PipelineModel.load(cat_pipeline_path)
            self.logger.info(f"✅ Categorical pipeline loaded")
            
            # Load vector assembler
            assembler_path = f"{self.preprocessing_path}/vector_assembler"
            components['vector_assembler'] = VectorAssembler.load(assembler_path)
            self.logger.info(f"✅ Vector assembler loaded")
            
            # Load standard scaler
            scaler_path = f"{self.preprocessing_path}/standard_scaler"
            components['standard_scaler'] = StandardScalerModel.load(scaler_path)
            self.logger.info(f"✅ Standard scaler loaded")
            
            # Load feature metadata
            metadata_path = f"{self.preprocessing_path}/feature_metadata.json"
            with open(metadata_path, 'r') as f:
                components['feature_metadata'] = json.load(f)
            self.logger.info(f"✅ Feature metadata loaded")
            
            self.logger.info("🎯 All components loaded successfully!")
            return components
            
        except Exception as e:
            self.logger.error(f"❌ Error loading components: {str(e)}")
            raise
    
    def validate_components(self, components: Dict[str, Any]) -> bool:
        """
        Validate that all required components are present and compatible
        
        Args:
            components: Dictionary of loaded components
        
        Returns:
            bool: True if all components are valid
        """
        required_components = [
            'model', 'categorical_pipeline', 'vector_assembler', 
            'standard_scaler', 'feature_metadata', 'manifest'
        ]
        
        self.logger.info("🔍 Validating components...")
        
        try:
            # Check all required components are present
            missing = [comp for comp in required_components if comp not in components]
            if missing:
                self.logger.error(f"❌ Missing components: {missing}")
                return False
            
            # Check metadata consistency
            metadata = components['feature_metadata']
            manifest = components['manifest']
            
            if metadata.get('model_name') != manifest.get('model_name'):
                self.logger.error("❌ Model name mismatch between metadata and manifest")
                return False
            
            self.logger.info("✅ All components validated successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error validating components: {str(e)}")
            return False
    
    def get_component_info(self) -> Dict[str, Any]:
        """
        Get information about saved components without loading them
        
        Returns:
            Dictionary containing component information
        """
        try:
            manifest_path = f"{self.preprocessing_path}/component_manifest.json"
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            metadata_path = f"{self.preprocessing_path}/feature_metadata.json"
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            return {
                "manifest": manifest,
                "metadata": metadata,
                "available": True
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "available": False
            }


# Convenience functions for direct use
def save_fraud_model_components(base_path: str,
                               model: LogisticRegressionModel,
                               categorical_pipeline: Pipeline,
                               vector_assembler: VectorAssembler,
                               scaler_model: StandardScalerModel,
                               feature_metadata: Dict[str, Any],
                               model_name: str = "fraud_model_june") -> Dict[str, str]:
    """
    Convenience function to save all fraud model components
    """
    manager = ModelComponentManager(base_path, model_name)
    return manager.save_all_components(
        model, categorical_pipeline, vector_assembler, 
        scaler_model, feature_metadata
    )


def load_fraud_model_components(base_path: str, 
                               model_name: str = "fraud_model_june") -> Dict[str, Any]:
    """
    Convenience function to load all fraud model components
    """
    manager = ModelComponentManager(base_path, model_name)
    return manager.load_all_components()


if __name__ == "__main__":
    # Example usage
    print("🔧 Model Component Management Utilities")
    print("Use this module to save and load fraud detection model components")
    print("\nExample usage:")
    print("from model_utils import save_fraud_model_components, load_fraud_model_components")
    print("components = load_fraud_model_components('/path/to/models')")