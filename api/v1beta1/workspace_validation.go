// Copyright (c) KAITO authors.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package v1beta1

import (
	"context"
	"fmt"
	"os"
	"reflect"
	"sort"
	"strconv"
	"strings"

	"github.com/distribution/reference"
	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/klog/v2"
	"knative.dev/pkg/apis"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"github.com/kaito-project/kaito/pkg/featuregates"
	"github.com/kaito-project/kaito/pkg/k8sclient"
	"github.com/kaito-project/kaito/pkg/model"
	"github.com/kaito-project/kaito/pkg/utils"
	"github.com/kaito-project/kaito/pkg/utils/consts"
	"github.com/kaito-project/kaito/pkg/utils/plugin"
)

const (
	N_SERIES_PREFIX = "Standard_N"
	D_SERIES_PREFIX = "Standard_D"

	DefaultLoraConfigMapTemplate   = "lora-params-template"
	DefaultQloraConfigMapTemplate  = "qlora-params-template"
	DefaultInferenceConfigTemplate = "inference-params-template"
	MaxAdaptersNumber              = 10

	// NVIDIA GPU labels for extracting GPU information
	NvidiaGPUCountLabel  = "nvidia.com/gpu.count"
	NvidiaGPUMemoryLabel = "nvidia.com/gpu.memory"
)

// GPUNodeType represents a group of nodes with identical GPU features
type GPUNodeType struct {
	// GPUProduct is the GPU model/product name.
	GPUProduct string
	// GPUCount is the number of GPUs per node.
	GPUCount int64
	// GPUMemory is the memory per GPU.
	GPUMemory resource.Quantity
	// NodeCount is the number of nodes in this bucket
	NodeCount int
	// NodeNames contains the names of nodes in this bucket
	NodeNames []string
}

// extractGPUFeaturesFromNode extracts GPU features from a single node using nvidia.com labels
func extractGPUFeaturesFromNode(node *corev1.Node) (*GPUNodeType, error) {
	var gpuProduct string
	var gpuCount int64
	var gpuMemoryMB int64

	// Extract GPU product
	if product, exists := node.Labels["nvidia.com/gpu.product"]; exists {
		gpuProduct = product
	}

	// Extract GPU count
	if countStr, exists := node.Labels[NvidiaGPUCountLabel]; exists {
		count, err := strconv.ParseInt(countStr, 10, 64)
		if err != nil {
			return nil, fmt.Errorf("invalid GPU count label: %v", err)
		}
		gpuCount = count
	} else {
		return nil, fmt.Errorf("missing nvidia.com/gpu.count label")
	}

	// Extract GPU memory per GPU (in MB)
	if memStr, exists := node.Labels[NvidiaGPUMemoryLabel]; exists {
		memMB, err := strconv.ParseInt(memStr, 10, 64)
		if err != nil {
			return nil, fmt.Errorf("invalid GPU memory label: %v", err)
		}
		gpuMemoryMB = memMB
	} else {
		return nil, fmt.Errorf("missing nvidia.com/gpu.memory label")
	}

	// Convert memory from MB to resource.Quantity (bytes)
	gpuMemoryBytes := gpuMemoryMB * 1024 * 1024
	gpuMemory := resource.NewQuantity(gpuMemoryBytes, resource.BinarySI)

	return &GPUNodeType{
		GPUProduct: gpuProduct,
		GPUCount:   gpuCount,
		GPUMemory:  *gpuMemory,
		NodeCount:  1,
		NodeNames:  []string{node.Name},
	}, nil
}

// queryNodesWithGPUFeatures queries nodes matching the label selector and extracts their GPU features
func (r *ResourceSpec) queryNodesWithGPUFeatures(ctx context.Context) ([]GPUNodeType, error) {
	if k8sclient.Client == nil {
		return nil, fmt.Errorf("k8s client not available")
	}

	// Convert label selector to map
	selector, err := metav1.LabelSelectorAsSelector(r.LabelSelector)
	if err != nil {
		return nil, fmt.Errorf("invalid label selector: %v", err)
	}

	// List nodes matching the selector
	var nodeList corev1.NodeList
	listOptions := &client.ListOptions{
		LabelSelector: selector,
	}

	err = k8sclient.Client.List(ctx, &nodeList, listOptions)
	if err != nil {
		return nil, fmt.Errorf("failed to list nodes: %v", err)
	}

	if len(nodeList.Items) == 0 {
		return nil, fmt.Errorf("no nodes found matching label selector")
	}

	// Group nodes by GPU features
	bucketMap := make(map[string]*GPUNodeType)

	for _, node := range nodeList.Items {
		gpuBucket, err := extractGPUFeaturesFromNode(&node)
		if err != nil {
			klog.Warningf("Failed to extract GPU features from node %s: %v", node.Name, err)
			continue
		}

		// Create a unique key for bucketing based on GPU features
		key := fmt.Sprintf("%s-%d-%s", gpuBucket.GPUProduct, gpuBucket.GPUCount, gpuBucket.GPUMemory.String())

		if bucket, exists := bucketMap[key]; exists {
			bucket.NodeCount++
			bucket.NodeNames = append(bucket.NodeNames, node.Name)
		} else {
			bucketMap[key] = gpuBucket
		}
	}

	// Convert map to slice and sort by capacity (GPUs per node * GPU memory per GPU, then by total nodes)
	var buckets []GPUNodeType
	for _, bucket := range bucketMap {
		buckets = append(buckets, *bucket)
	}

	// Sort buckets by total GPU capacity (descending): GPUs per node * memory per GPU * node count
	sort.Slice(buckets, func(i, j int) bool {
		capacityI := buckets[i].GPUCount * buckets[i].GPUMemory.Value() * int64(buckets[i].NodeCount)
		capacityJ := buckets[j].GPUCount * buckets[j].GPUMemory.Value() * int64(buckets[j].NodeCount)

		if capacityI != capacityJ {
			return capacityI > capacityJ // Descending order
		}

		// If total capacity is equal, prefer buckets with more GPUs per node
		if buckets[i].GPUCount != buckets[j].GPUCount {
			return buckets[i].GPUCount > buckets[j].GPUCount
		}

		// If GPUs per node are equal, prefer buckets with more memory per GPU
		return buckets[i].GPUMemory.Cmp(buckets[j].GPUMemory) > 0
	})

	return buckets, nil
}

// canBucketSatisfyModel checks if a given bucket can satisfy the model requirements
func canBucketSatisfyModel(bucket *GPUNodeType, modelGPUCount resource.Quantity, modelTotalGPUMemory resource.Quantity) (bool, int, string) {
	// Calculate total GPUs available in this bucket
	totalAvailableGPUs := bucket.GPUCount * int64(bucket.NodeCount)
	totalAvailableGPUMem := bucket.GPUMemory.Value() * bucket.GPUCount * int64(bucket.NodeCount)

	// Check if we have enough GPUs
	if totalAvailableGPUs < modelGPUCount.Value() {
		return false, 0, fmt.Sprintf("insufficient GPUs: bucket has %d GPUs but model requires %d",
			totalAvailableGPUs, modelGPUCount.Value())
	}

	// Check if we have enough GPU memory
	if totalAvailableGPUMem < modelTotalGPUMemory.Value() {
		totalAvailableGPUMemGiB := totalAvailableGPUMem / consts.GiBToBytes
		modelTotalGPUMemoryGiB := modelTotalGPUMemory.Value() / consts.GiBToBytes
		return false, 0, fmt.Sprintf("insufficient GPU memory: bucket has %d Gi (%s bytes) but model requires %d Gi (%s bytes)",
			totalAvailableGPUMemGiB, resource.NewQuantity(totalAvailableGPUMem, resource.BinarySI).String(),
			modelTotalGPUMemoryGiB, modelTotalGPUMemory.String())
	}

	// Calculate minimum nodes needed based on GPU count and memory constraints
	nodesForGPUs := int((modelGPUCount.Value() + bucket.GPUCount - 1) / bucket.GPUCount) // Ceiling division
	memoryPerNode := bucket.GPUMemory.Value() * bucket.GPUCount
	nodesForMemory := int((modelTotalGPUMemory.Value() + memoryPerNode - 1) / memoryPerNode) // Ceiling division

	minNodesNeeded := nodesForGPUs
	if nodesForMemory > minNodesNeeded {
		minNodesNeeded = nodesForMemory
	}

	if minNodesNeeded > bucket.NodeCount {
		return false, minNodesNeeded, fmt.Sprintf("insufficient nodes: bucket has %d nodes but requires %d nodes",
			bucket.NodeCount, minNodesNeeded)
	}

	return true, minNodesNeeded, ""
}

func (w *Workspace) SupportedVerbs() []admissionregistrationv1.OperationType {
	return []admissionregistrationv1.OperationType{
		admissionregistrationv1.Create,
		admissionregistrationv1.Update,
	}
}

func (w *Workspace) Validate(ctx context.Context) (errs *apis.FieldError) {
	errmsgs := validation.IsDNS1123Label(w.Name)
	if len(errmsgs) > 0 {
		errs = errs.Also(apis.ErrInvalidValue(strings.Join(errmsgs, ", "), "name"))
	}

	// Note: When NAP is disabled, node validation is handled in validateCreateWithInference
	// based on label selector matching and GPU features, not preferred nodes list

	base := apis.GetBaseline(ctx)
	if base == nil {
		klog.InfoS("Validate creation", "workspace", fmt.Sprintf("%s/%s", w.Namespace, w.Name))
		errs = errs.Also(w.validateCreate().ViaField("spec"))
		if w.Inference != nil {
			// Check if the bypass resource checks annotation is set
			bypassResourceChecks := false
			if w.GetAnnotations() != nil {
				if _, exists := w.GetAnnotations()[AnnotationBypassResourceChecks]; exists {
					bypassResourceChecks = true
				}
			}

			runtime := GetWorkspaceRuntimeName(w)
			// TODO: Add Adapter Spec Validation - Including DataSource Validation for Adapter
			errs = errs.Also(
				w.Resource.validateCreateWithInference(ctx, w.Inference, bypassResourceChecks, runtime).ViaField("resource"),
				w.Inference.validateCreate(ctx, runtime).ViaField("inference"),
				w.validateInferenceConfig(ctx),
			)
		}
		if w.Tuning != nil {
			// TODO: Add validate resource based on Tuning Spec
			errs = errs.Also(w.Resource.validateCreateWithTuning(w.Tuning).ViaField("resource"),
				w.Tuning.validateCreate(ctx, w.Namespace).ViaField("tuning"))
		}
	} else {
		klog.InfoS("Validate update", "workspace", fmt.Sprintf("%s/%s", w.Namespace, w.Name))
		old := base.(*Workspace)
		errs = errs.Also(
			w.validateUpdate(old).ViaField("spec"),
			w.Resource.validateUpdate(&old.Resource).ViaField("resource"),
		)
		if w.Inference != nil {
			errs = errs.Also(w.Inference.validateUpdate(old.Inference).ViaField("inference"))
		}
		if w.Tuning != nil {
			errs = errs.Also(w.Tuning.validateUpdate(old.Tuning).ViaField("tuning"))
		}
	}
	return errs
}

func (w *Workspace) validateCreate() (errs *apis.FieldError) {
	if w.Inference == nil && w.Tuning == nil {
		errs = errs.Also(apis.ErrGeneric("Either Inference or Tuning must be specified, not neither", ""))
	}
	if w.Inference != nil && w.Tuning != nil {
		errs = errs.Also(apis.ErrGeneric("Either Inference or Tuning must be specified, but not both", ""))
	}
	return errs
}

func (w *Workspace) validateUpdate(old *Workspace) (errs *apis.FieldError) {
	if (old.Inference == nil && w.Inference != nil) || (old.Inference != nil && w.Inference == nil) {
		errs = errs.Also(apis.ErrGeneric("Inference field cannot be toggled once set", "inference"))
	}

	if (old.Tuning == nil && w.Tuning != nil) || (old.Tuning != nil && w.Tuning == nil) {
		errs = errs.Also(apis.ErrGeneric("Tuning field cannot be toggled once set", "tuning"))
	}
	return errs
}

func (r *AdapterSpec) validateCreateorUpdate() (errs *apis.FieldError) {
	if r.Source == nil {
		errs = errs.Also(apis.ErrMissingField("Source"))
	} else {
		errs = errs.Also(r.Source.validateCreate().ViaField("Adapters"))

		if r.Source.Name == "" {
			errs = errs.Also(apis.ErrMissingField("Name of Adapter field must be specified"))
		} else if errmsgs := validation.IsDNS1123Subdomain(r.Source.Name); len(errmsgs) > 0 {
			errs = errs.Also(apis.ErrInvalidValue(strings.Join(errmsgs, ", "), "adapters.source.name"))
		}
		if r.Source.Image == "" {
			errs = errs.Also(apis.ErrMissingField("Image of Adapter field must be specified"))
		}
		if r.Strength == nil {
			var defaultStrength = "1.0"
			r.Strength = &defaultStrength
		}
		strength, err := strconv.ParseFloat(*r.Strength, 64)
		if err != nil {
			errs = errs.Also(apis.ErrGeneric(fmt.Sprintf("Invalid strength value for Adapter '%s': %v", r.Source.Name, err), "adapter"))
		}
		if strength < 0 || strength > 1.0 {
			errs = errs.Also(apis.ErrGeneric(fmt.Sprintf("Strength value for Adapter '%s' must be between 0 and 1", r.Source.Name), "adapter"))
		}

	}
	return errs
}

func (r *TuningSpec) validateCreate(ctx context.Context, workspaceNamespace string) (errs *apis.FieldError) {
	methodLowerCase := strings.ToLower(string(r.Method))
	if methodLowerCase != string(TuningMethodLora) && methodLowerCase != string(TuningMethodQLora) {
		errs = errs.Also(apis.ErrInvalidValue(r.Method, "Method"))
	}
	if r.Config == "" {
		klog.InfoS("Tuning config not specified. Using default based on method.")
		releaseNamespace, err := utils.GetReleaseNamespace()
		if err != nil {
			errs = errs.Also(apis.ErrGeneric(fmt.Sprintf("Failed to determine release namespace: %v", err), "namespace"))
		}
		defaultConfigMapTemplateName := ""
		if methodLowerCase == string(TuningMethodLora) {
			defaultConfigMapTemplateName = DefaultLoraConfigMapTemplate
		} else if methodLowerCase == string(TuningMethodQLora) {
			defaultConfigMapTemplateName = DefaultQloraConfigMapTemplate
		}
		if err := r.validateConfigMap(ctx, releaseNamespace, methodLowerCase, defaultConfigMapTemplateName); err != nil {
			errs = errs.Also(apis.ErrGeneric(fmt.Sprintf("Failed to evaluate validateConfigMap: %v", err), "Config"))
		}
	} else {
		if err := r.validateConfigMap(ctx, workspaceNamespace, methodLowerCase, r.Config); err != nil {
			errs = errs.Also(apis.ErrGeneric(fmt.Sprintf("Failed to evaluate validateConfigMap: %v", err), "Config"))
		}
	}
	if r.Input == nil {
		errs = errs.Also(apis.ErrMissingField("Input"))
	} else {
		errs = errs.Also(r.Input.validateCreate().ViaField("Input"))
	}
	if r.Output == nil {
		errs = errs.Also(apis.ErrMissingField("Output"))
	} else {
		errs = errs.Also(r.Output.validateCreate().ViaField("Output"))
	}
	// Currently require a preset to specified, in future we can consider defining a template
	if r.Preset == nil {
		errs = errs.Also(apis.ErrMissingField("Preset"))
	} else if presetName := string(r.Preset.Name); !plugin.IsValidPreset(presetName) {
		errs = errs.Also(apis.ErrInvalidValue(fmt.Sprintf("Unsupported tuning preset name %s", presetName), "presetName"))
	}
	return errs
}

func (r *TuningSpec) validateUpdate(old *TuningSpec) (errs *apis.FieldError) {
	if r.Input == nil {
		errs = errs.Also(apis.ErrMissingField("Input"))
	} else {
		errs = errs.Also(r.Input.validateUpdate(old.Input, true).ViaField("Input"))
	}
	if r.Output == nil {
		errs = errs.Also(apis.ErrMissingField("Output"))
	} else {
		errs = errs.Also(r.Output.validateUpdate().ViaField("Output"))
	}
	if !reflect.DeepEqual(old.Preset, r.Preset) {
		errs = errs.Also(apis.ErrGeneric("Preset cannot be changed", "Preset"))
	}
	oldMethod, newMethod := strings.ToLower(string(old.Method)), strings.ToLower(string(r.Method))
	if !reflect.DeepEqual(oldMethod, newMethod) {
		errs = errs.Also(apis.ErrGeneric("Method cannot be changed", "Method"))
	}
	// Consider supporting config fields changing
	return errs
}

func (r *DataSource) validateCreate() (errs *apis.FieldError) {
	sourcesSpecified := 0
	if len(r.URLs) > 0 {
		sourcesSpecified++
	}
	if image := r.Image; image != "" {
		if _, err := reference.ParseDockerRef(image); err != nil {
			errs = errs.Also(apis.ErrInvalidValue(fmt.Sprintf("Unable to parse input image reference: %s", err), "Image"))
		}

		sourcesSpecified++
	}

	if volume := r.Volume; volume != nil {
		sourcesSpecified++
	}

	// Ensure exactly one of URLs, Volume, or Image is specified
	if sourcesSpecified != 1 {
		errs = errs.Also(apis.ErrGeneric("Exactly one of URLs, Volume, or Image must be specified", "URLs", "Volume", "Image"))
	}

	return errs
}

func (r *DataSource) validateUpdate(old *DataSource, isTuning bool) (errs *apis.FieldError) {
	if isTuning && !reflect.DeepEqual(old.Name, r.Name) {
		errs = errs.Also(apis.ErrInvalidValue("During tuning Name field cannot be changed once set", "Name"))
	}
	if image := r.Image; image != "" {
		if _, err := reference.ParseDockerRef(image); err != nil {
			errs = errs.Also(apis.ErrInvalidValue(fmt.Sprintf("Unable to parse input image reference: %s", err), "Image"))
		}
	}

	return errs
}

func (r *DataDestination) validateCreate() (errs *apis.FieldError) {
	destinationsSpecified := 0
	if image := r.Image; image != "" {
		if _, err := reference.ParseDockerRef(image); err != nil {
			errs = errs.Also(apis.ErrInvalidValue(fmt.Sprintf("Unable to parse output image reference: %s", err), "Image"))
		}

		// Cloud Provider requires credentials to push image
		if r.ImagePushSecret == "" {
			errs = errs.Also(apis.ErrMissingField("Must specify imagePushSecret with destination image"))
		}

		destinationsSpecified++
	}

	if volume := r.Volume; volume != nil {
		destinationsSpecified++
	}

	// Ensure exactly one of Volume or Image is specified
	if destinationsSpecified != 1 {
		errs = errs.Also(apis.ErrMissingField("Exactly one of Volume or Image must be specified")) // TODO: Consider allowing both Volume and Image to be specified
	}
	return errs
}

func (r *DataDestination) validateUpdate() (errs *apis.FieldError) {
	if image := r.Image; image != "" {
		if _, err := reference.ParseDockerRef(image); err != nil {
			errs = errs.Also(apis.ErrInvalidValue(fmt.Sprintf("Unable to parse output image reference: %s", err), "Image"))
		}
	}

	return errs
}

func (r *ResourceSpec) validateCreateWithTuning(tuning *TuningSpec) (errs *apis.FieldError) {
	if *r.Count > 1 {
		errs = errs.Also(apis.ErrInvalidValue("Tuning does not currently support multinode configurations. Please set the node count to 1. Future support with DeepSpeed will allow this.", "count"))
	}
	return errs
}

func (r *ResourceSpec) validateCreateWithInference(ctx context.Context, inference *InferenceSpec, bypassResourceChecks bool, runtime model.RuntimeName) (errs *apis.FieldError) {
	var presetName string
	if inference.Preset != nil {
		presetName = strings.ToLower(string(inference.Preset.Name))
		// Since inference.Preset exists, we must validate preset name.
		if !plugin.IsValidPreset(presetName) {
			// Return to skip the rest of checks, the Inference spec validation will return proper err msg.
			return errs
		}
	}
	instanceType := string(r.InstanceType)
	// When NAP is disabled, validate using nvidia.com labels from nodes matching label selector instead of SKU lookup
	if featuregates.FeatureGates[consts.FeatureFlagDisableNodeAutoProvisioning] {
		if instanceType != "" {
			// TODO: we can change this to be more flexible.
			errs = errs.Also(apis.ErrInvalidValue("Instance type should not be specified when NAP is disabled. Use label selector to target nodes with nvidia.com labels instead", "instanceType"))
		}

		if presetName != "" {
			// Query nodes matching label selector and group by GPU features
			buckets, err := r.queryNodesWithGPUFeatures(ctx)
			if err != nil {
				errs = errs.Also(apis.ErrGeneric(fmt.Sprintf("Failed to query nodes with GPU features: %v", err), "labelSelector"))
			} else if len(buckets) == 0 {
				errs = errs.Also(apis.ErrGeneric("No nodes found with valid GPU features matching label selector", "labelSelector"))
			} else {
				modelPreset := plugin.KaitoModelRegister.MustGet(presetName)
				params := modelPreset.GetInferenceParameters()

				modelGPUCount := resource.MustParse(params.GPUCountRequirement)
				modelTotalGPUMemory := resource.MustParse(params.TotalSafeTensorFileSize)

				// Try to find a bucket that can satisfy the model requirements
				var selectedBucket *GPUNodeType
				var nodesNeeded int
				for i := range buckets {
					canSatisfy, nodes, _ := canBucketSatisfyModel(&buckets[i], modelGPUCount, modelTotalGPUMemory)
					if canSatisfy {
						selectedBucket = &buckets[i]
						nodesNeeded = nodes
						break
					}
				}

				if selectedBucket == nil {
					// No bucket can satisfy the requirements, build detailed error message
					var errorMessages []string
					errorMessages = append(errorMessages, fmt.Sprintf("No suitable node configuration found for preset %s", presetName))
					errorMessages = append(errorMessages, fmt.Sprintf("Model requires: %d GPUs, %s total GPU memory",
						modelGPUCount.Value(), modelTotalGPUMemory.String()))

					for i, bucket := range buckets {
						_, _, reason := canBucketSatisfyModel(&bucket, modelGPUCount, modelTotalGPUMemory)
						errorMessages = append(errorMessages, fmt.Sprintf("Bucket %d (%s, %d GPUs/node, %s/GPU, %d nodes): %s",
							i+1, bucket.GPUProduct, bucket.GPUCount, bucket.GPUMemory.String(), bucket.NodeCount, reason))
					}

					if bypassResourceChecks {
						klog.Warningf("Bypassing resource check: %s", strings.Join(errorMessages, "; "))
					} else {
						errs = errs.Also(apis.ErrInvalidValue(strings.Join(errorMessages, "; "), "labelSelector"))
					}
				} else {
					// Log successful match
					klog.Infof("Selected GPU node bucket for preset %s: %s with %d GPUs/node, %s/GPU, using %d out of %d available nodes",
						presetName, selectedBucket.GPUProduct, selectedBucket.GPUCount,
						selectedBucket.GPUMemory.String(), nodesNeeded, selectedBucket.NodeCount)
				}
			}
		}

		// Validate labelSelector
		if _, err := metav1.LabelSelectorAsMap(r.LabelSelector); err != nil {
			errs = errs.Also(apis.ErrInvalidValue(err.Error(), "labelSelector"))
		}

		return errs
	} else {
		// NAP is enabled - instance type is required
		if instanceType == "" {
			errs = errs.Also(apis.ErrMissingField("Instance type is required when node auto-provisioning is enabled"))
			return errs
		}

		// NAP is enabled - proceed with normal SKU-based validation
		skuHandler, err := utils.GetSKUHandler()
		if err != nil {
			errs = errs.Also(apis.ErrGeneric(fmt.Sprintf("Failed to get SKU handler: %v", err), "instanceType"))
			return errs
		}

		// Check if instancetype exists in our SKUs map for the particular cloud provider
		if skuConfig := skuHandler.GetGPUConfigBySKU(instanceType); skuConfig != nil {
			if presetName != "" {
				modelPreset := plugin.KaitoModelRegister.MustGet(presetName) // InferenceSpec has been validated so the name is valid.
				params := modelPreset.GetInferenceParameters()

				machineCount := *r.Count
				machineTotalNumGPUs := resource.NewQuantity(int64(machineCount*skuConfig.GPUCount), resource.DecimalSI)
				machineTotalGPUMem := resource.NewQuantity(int64(machineCount*skuConfig.GPUMemGB)*consts.GiBToBytes, resource.BinarySI) // Total GPU memory

				modelGPUCount := resource.MustParse(params.GPUCountRequirement)
				modelTotalGPUMemory := resource.MustParse(params.TotalSafeTensorFileSize)

				// Separate the checks for specific error messages
				if machineTotalNumGPUs.Cmp(modelGPUCount) < 0 {
					if bypassResourceChecks {
						klog.Warningf("Bypassing resource check: Insufficient number of GPUs detected but continuing due to bypass flag. Instance type %s provides %s, but preset %s requires at least %d",
							instanceType, machineTotalNumGPUs.String(), presetName, modelGPUCount.Value())
					} else {
						errs = errs.Also(apis.ErrInvalidValue(
							fmt.Sprintf(
								"Insufficient number of GPUs: Instance type %s provides %s, but preset %s requires at least %d",
								instanceType,
								machineTotalNumGPUs.String(),
								presetName,
								modelGPUCount.Value(),
							),
							"instanceType",
						))
					}
				}

				if machineTotalGPUMem.Cmp(modelTotalGPUMemory) < 0 {
					// Convert to GiB for readable error messages
					machineTotalGPUMemGiB := machineTotalGPUMem.Value() / consts.GiBToBytes
					modelTotalGPUMemoryGiB := modelTotalGPUMemory.Value() / consts.GiBToBytes

					if bypassResourceChecks {
						klog.Warningf("Bypassing resource check: Insufficient total GPU memory detected but continuing due to bypass flag. Instance type %s has a total of %d Gi (%s bytes), but preset %s requires at least %d Gi (%s bytes)",
							instanceType, machineTotalGPUMemGiB, machineTotalGPUMem.String(), presetName, modelTotalGPUMemoryGiB, modelTotalGPUMemory.String())
					} else {
						errs = errs.Also(apis.ErrInvalidValue(
							fmt.Sprintf(
								"Insufficient total GPU memory: Instance type %s has a total of %d Gi (%s bytes), but preset %s requires at least %d Gi (%s bytes)",
								instanceType,
								machineTotalGPUMemGiB,
								machineTotalGPUMem.String(),
								presetName,
								modelTotalGPUMemoryGiB,
								modelTotalGPUMemory.String(),
							),
							"instanceType",
						))
					}
				}

				// If the model preset supports distributed inference, and a single machine has insufficient GPU memory to run the model,
				// then we need to make sure the Workspace is not using the Huggingface Transformers runtime since it no longer supports
				// multi-node distributed inference.
				totalGPUMemoryPerMachine := resource.NewQuantity(int64(skuConfig.GPUMemGB)*consts.GiBToBytes, resource.BinarySI)
				distributedInferenceRequired := modelTotalGPUMemory.Cmp(*totalGPUMemoryPerMachine) > 0
				if modelPreset.SupportDistributedInference() && distributedInferenceRequired && runtime == model.RuntimeNameHuggingfaceTransformers {
					errs = errs.Also(apis.ErrGeneric("Multi-node distributed inference is not supported with Huggingface Transformers runtime"))
				}
			}
		} else {
			provider := os.Getenv("CLOUD_PROVIDER")
			// Check for other instance types pattern matches if cloud provider is Azure
			if provider != consts.AzureCloudName || (!strings.HasPrefix(instanceType, N_SERIES_PREFIX) && !strings.HasPrefix(instanceType, D_SERIES_PREFIX)) {
				errs = errs.Also(apis.ErrInvalidValue(fmt.Sprintf("Instance type %s is unsupported with node auto-provisioning. Disable auto-provisioning feature flag or switch to a supported SKU: %s", instanceType, skuHandler.GetSupportedSKUs()), "instanceType"))
			}
		}
	}

	// Validate labelSelector
	if _, err := metav1.LabelSelectorAsMap(r.LabelSelector); err != nil {
		errs = errs.Also(apis.ErrInvalidValue(err.Error(), "labelSelector"))
	}

	return errs
}

func (r *ResourceSpec) validateUpdate(old *ResourceSpec) (errs *apis.FieldError) {
	// We disable changing node count for now.
	if r.Count != nil && old.Count != nil && *r.Count != *old.Count {
		errs = errs.Also(apis.ErrGeneric("field is immutable", "count"))
	}
	if r.InstanceType != old.InstanceType {
		errs = errs.Also(apis.ErrGeneric("field is immutable", "instanceType"))
	}
	newLabels, err0 := metav1.LabelSelectorAsMap(r.LabelSelector)
	oldLabels, err1 := metav1.LabelSelectorAsMap(old.LabelSelector)
	if err0 != nil || err1 != nil {
		errs = errs.Also(apis.ErrGeneric("Only allow matchLabels or 'IN' matchExpression", "labelSelector"))
	} else {
		if !reflect.DeepEqual(newLabels, oldLabels) {
			errs = errs.Also(apis.ErrGeneric("field is immutable", "labelSelector"))
		}
	}
	return errs
}

func (i *InferenceSpec) validateCreate(ctx context.Context, runtime model.RuntimeName) (errs *apis.FieldError) {
	// Check if both Preset and Template are not set
	if i.Preset == nil && i.Template == nil {
		errs = errs.Also(apis.ErrMissingField("Preset or Template must be specified"))
	}

	// Check if both Preset and Template are set at the same time
	if i.Preset != nil && i.Template != nil {
		errs = errs.Also(apis.ErrGeneric("Preset and Template cannot be set at the same time"))
	}

	if i.Preset != nil {
		presetName := string(i.Preset.Name)
		// Validate preset name
		if !plugin.IsValidPreset(presetName) {
			errs = errs.Also(apis.ErrInvalidValue(fmt.Sprintf("Unsupported inference preset name %s", presetName), "presetName"))
			// Need to return here. Otherwise, a panic will be hit when doing following checks.
			return errs
		}
		modelPreset := plugin.KaitoModelRegister.MustGet(string(i.Preset.Name))
		params := modelPreset.GetInferenceParameters()
		useAdapterStrength := false
		for _, adapter := range i.Adapters {
			if adapter.Strength != nil {
				useAdapterStrength = true
				break
			}
		}
		err := params.Validate(model.RuntimeContext{
			RuntimeName: runtime,
			RuntimeContextExtraArguments: model.RuntimeContextExtraArguments{
				AdaptersEnabled:        len(i.Adapters) > 0,
				AdapterStrengthEnabled: useAdapterStrength,
			},
		})
		if err != nil {
			errs = errs.Also(apis.ErrGeneric(fmt.Sprintf("Runtime validation: %v", err)))
		}
		// For models that require downloading at runtime, we need to check if the modelAccessSecret is provided
		if params.DownloadAtRuntime && i.Preset.PresetOptions.ModelAccessSecret == "" {
			errs = errs.Also(apis.ErrGeneric("This preset requires a modelAccessSecret with HF_TOKEN key under presetOptions to download the model"))
		} else if !params.DownloadAtRuntime && i.Preset.PresetOptions.ModelAccessSecret != "" {
			errs = errs.Also(apis.ErrGeneric("This preset does not require a modelAccessSecret with HF_TOKEN key under presetOptions"))
		}
	}
	if len(i.Adapters) > MaxAdaptersNumber {
		errs = errs.Also(apis.ErrGeneric(fmt.Sprintf("Number of Adapters exceeds the maximum limit, maximum of %s allowed", strconv.Itoa(MaxAdaptersNumber))))
	}

	// check if adapter names are duplicate
	if len(i.Adapters) > 0 {
		nameMap := make(map[string]bool)
		errs = errs.Also(validateDuplicateName(i.Adapters, nameMap))
	}

	return errs
}

func (i *InferenceSpec) validateUpdate(old *InferenceSpec) (errs *apis.FieldError) {
	if !reflect.DeepEqual(i.Preset, old.Preset) {
		errs = errs.Also(apis.ErrGeneric("field is immutable", "preset"))
	}
	// inference.template can be changed, but cannot be set/unset.
	if (i.Template != nil && old.Template == nil) || (i.Template == nil && old.Template != nil) {
		errs = errs.Also(apis.ErrGeneric("field cannot be unset/set if it was set/unset", "template"))
	}

	// check if adapter names are duplicate
	for _, adapter := range i.Adapters {
		errs = errs.Also(adapter.validateCreateorUpdate())
	}

	// check if adapter names are duplicate

	if len(i.Adapters) > 0 {
		nameMap := make(map[string]bool)
		errs = errs.Also(validateDuplicateName(i.Adapters, nameMap))
	}
	return errs
}

func validateDuplicateName(adapters []AdapterSpec, nameMap map[string]bool) (errs *apis.FieldError) {
	for _, adapter := range adapters {
		if _, ok := nameMap[adapter.Source.Name]; ok {
			errs = errs.Also(apis.ErrGeneric(fmt.Sprintf("Duplicate adapter source name found: %s", adapter.Source.Name)))
		} else {
			nameMap[adapter.Source.Name] = true
		}
	}
	return errs
}
