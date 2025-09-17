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

package controllers

import (
	"context"
	"fmt"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/klog/v2"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"

	kaitov1beta1 "github.com/kaito-project/kaito/api/v1beta1"
	"github.com/kaito-project/kaito/pkg/featuregates"
	"github.com/kaito-project/kaito/pkg/utils/consts"
	"github.com/kaito-project/kaito/pkg/utils/nodeclaim"
	"github.com/kaito-project/kaito/pkg/utils/workspace"
)

// garbageCollectWorkspace remove finalizer associated with workspace object.
func (c *WorkspaceReconciler) garbageCollectWorkspace(ctx context.Context, wObj *kaitov1beta1.Workspace) (ctrl.Result, error) {
	klog.InfoS("garbageCollectWorkspace", "workspace", klog.KObj(wObj))

	// Remove workspace labels from nodes when NAP is disabled
	if featuregates.FeatureGates[consts.FeatureFlagDisableNodeAutoProvisioning] {
		if err := c.removeWorkspaceLabelsFromNodes(ctx, wObj); err != nil {
			klog.ErrorS(err, "failed to remove workspace labels from nodes", "workspace", klog.KObj(wObj))
			return ctrl.Result{}, err
		}
	}

	// Check if there are any nodeClaims associated with this workspace.
	ncList, err := nodeclaim.ListNodeClaim(ctx, wObj, c.Client)
	if err != nil {
		return ctrl.Result{}, err
	}

	// We should delete all the nodeClaims that are created by this workspace
	for i := range ncList.Items {
		if ncList.Items[i].DeletionTimestamp.IsZero() {
			klog.InfoS("Deleting associated NodeClaim...", "nodeClaim", ncList.Items[i].Name)
			if deleteErr := c.Delete(ctx, &ncList.Items[i], &client.DeleteOptions{}); deleteErr != nil {
				klog.ErrorS(deleteErr, "failed to delete the nodeClaim", "nodeClaim", klog.KObj(&ncList.Items[i]))
				return ctrl.Result{}, deleteErr
			}
		}
	}

	updateErr := workspace.UpdateWorkspaceWithRetry(ctx, c.Client, wObj, func(ws *kaitov1beta1.Workspace) error {
		controllerutil.RemoveFinalizer(ws, consts.WorkspaceFinalizer)
		return nil
	})
	if updateErr != nil {
		if apierrors.IsNotFound(updateErr) {
			return ctrl.Result{}, nil
		}
		klog.ErrorS(updateErr, "failed to update the workspace to remove finalizer", "workspace", klog.KObj(wObj))
		return ctrl.Result{}, updateErr
	}

	klog.InfoS("successfully removed the workspace finalizers", "workspace", klog.KObj(wObj))

	return ctrl.Result{}, nil
}

// removeWorkspaceLabelsFromNodes removes workspace labels from all nodes associated with the workspace
func (c *WorkspaceReconciler) removeWorkspaceLabelsFromNodes(ctx context.Context, wObj *kaitov1beta1.Workspace) error {
	// List all nodes with this workspace's label
	nodeList := &corev1.NodeList{}
	err := c.Client.List(ctx, nodeList, client.MatchingLabels{WorkspaceNameLabel: wObj.Name})
	if err != nil {
		return fmt.Errorf("failed to list nodes with workspace label: %w", err)
	}

	// Remove workspace label from each node
	for i := range nodeList.Items {
		node := &nodeList.Items[i]
		if _, exists := node.Labels[WorkspaceNameLabel]; exists {
			patch := client.MergeFrom(node.DeepCopy())
			delete(node.Labels, WorkspaceNameLabel)

			if err := c.Client.Patch(ctx, node, patch); err != nil {
				return fmt.Errorf("failed to remove workspace label from node %s: %w", node.Name, err)
			}
			klog.InfoS("Removed workspace label from node", "node", node.Name, "workspace", wObj.Name)
		}
	}

	return nil
}
