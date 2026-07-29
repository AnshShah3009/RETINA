pub mod types;
pub mod ops;

pub use types::*;
pub use ops::*;

#[cfg(test)]
use crate::storage::{CpuStorage, Storage};

#[cfg(test)]
mod tests {
    use super::*;

    mod tensor_shape_tests {
        use super::*;

        #[test]
        fn test_tensor_shape_new() {
            let shape = TensorShape::new(3, 100, 200);

            assert_eq!(shape.channels, 3);
            assert_eq!(shape.height, 100);
            assert_eq!(shape.width, 200);
        }

        #[test]
        fn test_tensor_shape_len() {
            let shape = TensorShape::new(3, 100, 200);
            assert_eq!(shape.len(), 3 * 100 * 200);
        }

        #[test]
        fn test_tensor_shape_hw() {
            let shape = TensorShape::new(3, 100, 200);
            let (h, w) = shape.hw();
            assert_eq!(h, 100);
            assert_eq!(w, 200);
        }

        #[test]
        fn test_tensor_shape_chw() {
            let shape = TensorShape::new(3, 100, 200);
            let (c, h, w) = shape.chw();
            assert_eq!(c, 3);
            assert_eq!(h, 100);
            assert_eq!(w, 200);
        }

        #[test]
        fn test_tensor_shape_is_1d() {
            assert!(TensorShape::new(100, 1, 1).is_1d());
            assert!(!TensorShape::new(1, 100, 1).is_1d());
        }

        #[test]
        fn test_tensor_shape_is_2d() {
            assert!(TensorShape::new(1, 100, 200).is_2d());
            assert!(!TensorShape::new(3, 100, 200).is_2d());
        }

        #[test]
        fn test_tensor_shape_is_3d() {
            assert!(TensorShape::new(3, 100, 200).is_3d());
        }
    }

    mod tensor_creation_tests {
        use super::*;

        #[test]
        fn test_tensor_zeros() {
            let tensor: Tensor<f32> = Tensor::zeros(TensorShape::new(1, 10, 10)).unwrap();

            assert_eq!(tensor.shape.channels, 1);
            assert_eq!(tensor.shape.height, 10);
            assert_eq!(tensor.shape.width, 10);
        }

        #[test]
        fn test_tensor_ones() {
            let tensor: Tensor<f32> = Tensor::ones(TensorShape::new(1, 2, 2)).unwrap();

            let slice = tensor.as_slice().unwrap();
            for &v in slice {
                assert!((v - 1.0).abs() < 1e-5);
            }
        }

        #[test]
        fn test_tensor_from_vec() {
            let data = vec![1.0f32, 2.0, 3.0, 4.0];
            let tensor: Tensor<f32> =
                Tensor::from_vec(data.clone(), TensorShape::new(1, 2, 2)).unwrap();

            let slice = tensor.as_slice().unwrap();
            assert_eq!(slice, &data[..]);
        }

        #[test]
        fn test_tensor_from_vec_wrong_size() {
            let data = vec![1.0f32, 2.0, 3.0];
            let result: Result<Tensor<f32>, _> = Tensor::from_vec(data, TensorShape::new(1, 2, 2));

            assert!(result.is_err());
        }
    }

    mod tensor_slice_tests {
        use super::*;

        #[test]
        fn test_slice_valid() {
            let tensor: Tensor<f32> = Tensor::zeros(TensorShape::new(1, 10, 10)).unwrap();

            let slice = tensor.slice(0..1, 2..5, 3..7).unwrap();
            assert_eq!(slice.shape.height, 3);
            assert_eq!(slice.shape.width, 4);
        }

        #[test]
        fn test_slice_out_of_bounds() {
            let tensor: Tensor<f32> = Tensor::zeros(TensorShape::new(1, 10, 10)).unwrap();

            let result = tensor.slice(0..1, 0..20, 0..10);
            assert!(result.is_err());
        }

        #[test]
        fn test_slice_empty() {
            let tensor: Tensor<f32> = Tensor::zeros(TensorShape::new(1, 10, 10)).unwrap();

            let result = tensor.slice(0..1, 5..5, 0..10);
            assert!(result.is_ok());
        }

        #[test]
        fn test_slice_copy() {
            let mut tensor: Tensor<f32> = Tensor::zeros(TensorShape::new(1, 10, 10)).unwrap();

            {
                let data = tensor.as_mut_slice().unwrap();
                for v in data.iter_mut() {
                    *v = 42.0;
                }
            }

            let slice = tensor.slice(0..1, 2..5, 3..7).unwrap();
            let slice_data = slice.as_slice().unwrap();
            for &v in slice_data {
                assert!((v - 42.0).abs() < 1e-5);
            }
        }
    }

    mod tensor_concat_tests {
        use super::*;

        fn create_tensor_with_value(shape: TensorShape, val: f32) -> Tensor<f32> {
            let mut t = Tensor::zeros(shape).unwrap();
            let slice = t.as_mut_slice().unwrap();
            for v in slice.iter_mut() {
                *v = val;
            }
            t
        }

        #[test]
        fn test_concat_along_channels() {
            let t1 = create_tensor_with_value(TensorShape::new(1, 2, 2), 1.0);
            let t2 = create_tensor_with_value(TensorShape::new(2, 2, 2), 2.0);

            let result = Tensor::concat(&[&t1, &t2], 0).unwrap();

            assert_eq!(result.shape.channels, 3);
            assert_eq!(result.shape.height, 2);
            assert_eq!(result.shape.width, 2);
        }

        #[test]
        fn test_concat_along_height() {
            let t1 = create_tensor_with_value(TensorShape::new(1, 2, 4), 1.0);
            let t2 = create_tensor_with_value(TensorShape::new(1, 3, 4), 2.0);

            let result = Tensor::concat(&[&t1, &t2], 1).unwrap();

            assert_eq!(result.shape.channels, 1);
            assert_eq!(result.shape.height, 5);
            assert_eq!(result.shape.width, 4);
        }

        #[test]
        fn test_concat_along_width() {
            let t1 = create_tensor_with_value(TensorShape::new(1, 2, 3), 1.0);
            let t2 = create_tensor_with_value(TensorShape::new(1, 2, 4), 2.0);

            let result = Tensor::concat(&[&t1, &t2], 2).unwrap();

            assert_eq!(result.shape.channels, 1);
            assert_eq!(result.shape.height, 2);
            assert_eq!(result.shape.width, 7);
        }

        #[test]
        fn test_concat_empty_list() {
            let tensors: Vec<&Tensor<f32>> = vec![];

            let result = Tensor::concat(&tensors, 0);
            assert!(result.is_err());
        }

        #[test]
        fn test_concat_shape_mismatch() {
            let t1 = create_tensor_with_value(TensorShape::new(1, 2, 3), 1.0);
            let t2 = create_tensor_with_value(TensorShape::new(1, 4, 5), 2.0);

            let result = Tensor::concat(&[&t1, &t2], 0);
            assert!(result.is_err());
        }

        #[test]
        fn test_concat_invalid_dim() {
            let t1 = create_tensor_with_value(TensorShape::new(1, 2, 2), 1.0);
            let t2 = create_tensor_with_value(TensorShape::new(1, 2, 2), 2.0);

            let result = Tensor::concat(&[&t1, &t2], 3);
            assert!(result.is_err());
        }
    }

    mod tensor_operations_tests {
        use super::*;

        #[test]
        fn test_index() {
            let mut tensor: Tensor<f32> = Tensor::zeros(TensorShape::new(3, 4, 5)).unwrap();

            {
                let slice = tensor.as_mut_slice().unwrap();
                slice[0] = 1.0;
                slice[1] = 2.0;
            }

            let idx0 = tensor.index(0, 0, 0).unwrap();
            let idx1 = tensor.index(0, 0, 1).unwrap();

            assert!((idx0 - 1.0).abs() < 1e-5);
            assert!((idx1 - 2.0).abs() < 1e-5);
        }

        #[test]
        fn test_index_out_of_bounds() {
            let tensor: Tensor<f32> = Tensor::zeros(TensorShape::new(1, 2, 2)).unwrap();

            let result = tensor.index(2, 0, 0);
            assert!(result.is_err());
        }

        #[test]
        fn test_index_mut() {
            let mut tensor: Tensor<f32> = Tensor::zeros(TensorShape::new(1, 2, 2)).unwrap();

            *tensor.index_mut(0, 0, 0).unwrap() = 5.0;

            assert!((tensor.index(0, 0, 0).unwrap() - 5.0).abs() < 1e-5);
        }

        #[test]
        fn test_reshape() {
            let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
            let tensor: Tensor<f32> = Tensor::from_vec(data, TensorShape::new(1, 2, 3)).unwrap();

            let reshaped = tensor.reshape(TensorShape::new(1, 3, 2)).unwrap();

            assert_eq!(reshaped.shape.height, 3);
            assert_eq!(reshaped.shape.width, 2);
        }
    }

    mod tensor_image_tests {
        use super::*;

        #[test]
        fn test_from_image_gray() {
            let data = vec![0u8, 128, 255];
            let result = Tensor::<f32>::from_image_gray(&data, 3, 1).unwrap();

            assert_eq!(result.shape.width, 3);
            assert_eq!(result.shape.height, 1);
            assert_eq!(result.shape.channels, 1);
        }

        #[test]
        fn test_from_image_gray_wrong_size() {
            let data = vec![0u8, 128];
            let result = Tensor::<f32>::from_image_gray(&data, 2, 2);

            assert!(result.is_err());
        }

        #[test]
        fn test_from_image_rgb() {
            let data = vec![255u8, 0, 0, 0, 255, 0];
            let result = Tensor::<f32>::from_image_rgb(&data, 1, 2).unwrap();

            assert_eq!(result.shape.width, 1);
            assert_eq!(result.shape.height, 2);
            assert_eq!(result.shape.channels, 3);
        }
    }

    mod helper_function_tests {
        use super::*;

        #[test]
        fn test_create_tensor_2d() {
            let tensor: Tensor<f32> = create_tensor_2d(10, 20).unwrap();

            assert_eq!(tensor.shape.channels, 1);
            assert_eq!(tensor.shape.height, 10);
            assert_eq!(tensor.shape.width, 20);
        }

        #[test]
        fn test_create_tensor_3d() {
            let tensor: Tensor<f32> = create_tensor_3d(3, 10, 20).unwrap();

            assert_eq!(tensor.shape.channels, 3);
            assert_eq!(tensor.shape.height, 10);
            assert_eq!(tensor.shape.width, 20);
        }
    }

    mod tensor_storage_generic_tests {
        use super::*;
        use crate::storage::DeviceType;

        #[test]
        fn test_tensor_cpu_storage_creation() {
            let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
            let shape = TensorShape::new(1, 2, 3);
            let tensor: Tensor<f32, CpuStorage<f32>> =
                Tensor::from_vec(data.clone(), shape).unwrap();

            assert_eq!(tensor.storage.len(), 6);
            assert_eq!(tensor.shape, shape);
            assert_eq!(tensor.as_slice().unwrap(), &data[..]);
        }

        #[test]
        fn test_tensor_storage_access() {
            let data = vec![1.0f32, 2.0, 3.0, 4.0];
            let shape = TensorShape::new(1, 2, 2);
            let mut tensor: Tensor<f32, CpuStorage<f32>> =
                Tensor::from_vec(data.clone(), shape).unwrap();

            // Test immutable access through storage
            let slice = tensor.as_slice().unwrap();
            assert_eq!(slice.len(), 4);

            // Test mutable access through storage
            let mut_slice = tensor.as_mut_slice().unwrap();
            mut_slice[0] = 10.0;

            // Verify mutation
            let updated = tensor.as_slice().unwrap();
            assert!((updated[0] - 10.0).abs() < 1e-5);
        }

        #[test]
        fn test_tensor_from_vec_cpu_storage() {
            let data = vec![5.0f32; 12];
            let shape = TensorShape::new(3, 2, 2);
            let tensor: Tensor<f32, CpuStorage<f32>> =
                Tensor::from_vec(data.clone(), shape).unwrap();

            // Verify the storage is CpuStorage
            assert_eq!(tensor.storage.len(), 12);
            assert_eq!(tensor.storage.device(), DeviceType::Cpu);
        }

        #[test]
        fn test_tensor_slice_access_f32() {
            let data = vec![1.0f32, 2.0, 3.0];
            let shape = TensorShape::new(1, 1, 3);
            let tensor: Tensor<f32, CpuStorage<f32>> =
                Tensor::from_vec(data.clone(), shape).unwrap();

            // Should be able to access slice through existing API
            let slice = tensor.as_slice().unwrap();
            assert_eq!(slice, &data[..]);
        }

        #[test]
        fn test_tensor_mut_slice_access_f32() {
            let data = vec![1.0f32, 2.0, 3.0];
            let shape = TensorShape::new(1, 1, 3);
            let mut tensor: Tensor<f32, CpuStorage<f32>> =
                Tensor::from_vec(data.clone(), shape).unwrap();

            // Should be able to access mutable slice through existing API
            {
                let mut_slice = tensor.as_mut_slice().unwrap();
                for v in mut_slice.iter_mut() {
                    *v *= 2.0;
                }
            }

            let updated = tensor.as_slice().unwrap();
            assert!((updated[0] - 2.0).abs() < 1e-5);
            assert!((updated[1] - 4.0).abs() < 1e-5);
            assert!((updated[2] - 6.0).abs() < 1e-5);
        }

        #[test]
        fn test_tensor_reshape_generic_storage() {
            let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
            let shape1 = TensorShape::new(1, 2, 3);
            let tensor: Tensor<f32, CpuStorage<f32>> =
                Tensor::from_vec(data.clone(), shape1).unwrap();

            let shape2 = TensorShape::new(1, 3, 2);
            let reshaped = tensor.reshape(shape2).unwrap();

            assert_eq!(reshaped.shape, shape2);
            assert_eq!(reshaped.as_slice().unwrap(), &data[..]);
        }

        #[test]
        fn test_data_type_is_float() {
            assert!(DataType::F32.is_float());
            assert!(DataType::F64.is_float());
            assert!(!DataType::U8.is_float());
            assert!(!DataType::I32.is_float());

            #[cfg(feature = "half-precision")]
            {
                assert!(DataType::F16.is_float());
                assert!(DataType::BF16.is_float());
            }
        }

        #[test]
        fn test_tensor_convert_precision() {
            let data = vec![1.0f32, 2.0, 3.0, 4.0];
            let shape = TensorShape::new(1, 2, 2);
            let tensor_f32: CpuTensor<f32> = CpuTensor::from_vec(data, shape).unwrap();

            let tensor_f64: CpuTensor<f64> = tensor_f32.convert_precision::<f64, _>().unwrap();

            assert_eq!(tensor_f64.shape, shape);
            assert_eq!(tensor_f64.dtype, DataType::F64);
            let slice_f64 = tensor_f64.as_slice().unwrap();
            assert!((slice_f64[0] - 1.0).abs() < 1e-10);
            assert!((slice_f64[3] - 4.0).abs() < 1e-10);

            #[cfg(feature = "half-precision")]
            {
                let tensor_f16: CpuTensor<half::f16> =
                    tensor_f32.convert_precision::<half::f16, _>().unwrap();
                assert_eq!(tensor_f16.dtype, DataType::F16);
                assert!((tensor_f16.as_slice().unwrap()[0].to_f32() - 1.0).abs() < 1e-3);
            }
        }

        #[test]
        fn test_tensor_storage_device_type() {
            let data = vec![1.0f32; 24];
            let shape = TensorShape::new(2, 3, 4);
            let tensor: Tensor<f32, CpuStorage<f32>> = Tensor::from_vec(data, shape).unwrap();

            // Verify the storage is CPU-based
            assert_eq!(tensor.storage.device(), DeviceType::Cpu);
        }

        #[test]
        fn test_tensor_storage_capacity() {
            let data = vec![1.0f32, 2.0, 3.0];
            let shape = TensorShape::new(1, 1, 3);
            let tensor: Tensor<f32, CpuStorage<f32>> = Tensor::from_vec(data, shape).unwrap();

            // Verify storage capacity matches data length
            assert_eq!(tensor.storage.capacity(), 3);
            assert_eq!(tensor.storage.len(), 3);
        }

        #[test]
        fn test_tensor_with_storage_conversion() {
            let data = vec![1.0f32, 2.0, 3.0, 4.0];
            let shape = TensorShape::new(1, 2, 2);
            let tensor1: Tensor<f32, CpuStorage<f32>> =
                Tensor::from_vec(data.clone(), shape).unwrap();

            // Create a new storage with same data
            let storage2: CpuStorage<f32> = CpuStorage::from_vec(data).unwrap();

            // Use with_storage to change storage
            let tensor2 = tensor1.with_storage(storage2);

            // Should have new storage but same shape and data
            assert_eq!(tensor2.shape, shape);
            assert_eq!(tensor2.as_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
        }
    }

    mod cpu_convenience_api_tests {
        use super::*;

        #[test]
        fn test_cpu_as_slice() {
            let data = vec![1.0f32, 2.0, 3.0, 4.0];
            let shape = TensorShape::new(1, 2, 2);
            let tensor = Tensor::<f32, CpuStorage<f32>>::from_vec(data.clone(), shape).unwrap();

            let slice = tensor.cpu_as_slice();
            assert_eq!(slice, &[1.0, 2.0, 3.0, 4.0]);
        }

        #[test]
        fn test_cpu_as_mut_slice() {
            let data = vec![1.0f32, 2.0, 3.0, 4.0];
            let shape = TensorShape::new(1, 2, 2);
            let mut tensor = Tensor::<f32, CpuStorage<f32>>::from_vec(data, shape).unwrap();

            let slice = tensor.cpu_as_mut_slice();
            slice[0] = 42.0;
            slice[1] = 99.0;

            assert_eq!(tensor.cpu_as_slice()[0], 42.0);
            assert_eq!(tensor.cpu_as_slice()[1], 99.0);
        }

        #[test]
        fn test_cpu_to_vec() {
            let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
            let shape = TensorShape::new(1, 1, 5);
            let tensor = Tensor::<f32, CpuStorage<f32>>::from_vec(data.clone(), shape).unwrap();

            let vec = tensor.cpu_to_vec();
            assert_eq!(vec, data);

            // Verify it's a clone, not the same reference
            assert_eq!(vec, tensor.cpu_as_slice());
        }

        #[test]
        fn test_cpu_to_vec_independence() {
            let data = vec![1.0f32, 2.0, 3.0];
            let shape = TensorShape::new(1, 1, 3);
            let tensor = Tensor::<f32, CpuStorage<f32>>::from_vec(data, shape).unwrap();

            let mut vec = tensor.cpu_to_vec();
            vec[0] = 999.0;

            // Modifying the vec should not affect the tensor
            assert_ne!(vec[0], tensor.cpu_as_slice()[0]);
            assert_eq!(tensor.cpu_as_slice()[0], 1.0);
        }

        #[test]
        fn test_cpu_convenience_with_different_types() {
            let data_i32 = vec![1i32, 2, 3, 4];
            let shape = TensorShape::new(2, 2, 1);
            let tensor = Tensor::<i32, CpuStorage<i32>>::from_vec(data_i32.clone(), shape).unwrap();

            assert_eq!(tensor.cpu_as_slice(), &[1, 2, 3, 4]);
            assert_eq!(tensor.cpu_to_vec(), data_i32);
        }
    }
}
