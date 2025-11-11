// Parameters
@description('Location for all resources')
param location string

@description('Environment name')
param environment string

@description('Resource prefix')
param resourcePrefix string

@description('Unique suffix for resource naming')
param uniqueSuffix string

@description('Virtual Network Resource ID')
param virtualNetworkResourceId string

@description('Private Endpoint Subnet Resource ID')
param privateEndpointSubnetResourceId string

@description('Key Vault Resource ID')
param keyVaultResourceId string

@description('Application Insights Resource ID')
param applicationInsightsResourceId string

@description('Storage Account Resource ID')
param storageAccountResourceId string

@description('Principal ID for role assignments')
param principalId string

@description('Principal type (User or ServicePrincipal)')
@allowed(['User', 'ServicePrincipal'])
param principalType string

// Variables
var mlWorkspaceName = '${resourcePrefix}-${environment}-mlw-${uniqueSuffix}'
var cognitiveSearchName = '${resourcePrefix}-${environment}-search-${uniqueSuffix}'

// Machine Learning Workspace using Azure Verified Module
module machineLearningWorkspace 'br/public:avm/res/machine-learning-services/workspace:0.7.0' = {
  name: 'machineLearningWorkspace'
  params: {
    name: mlWorkspaceName
    location: location
    sku: 'Basic'
    associatedStorageAccountResourceId: storageAccountResourceId
    associatedContainerRegistryResourceId: keyVaultResourceId
    associatedApplicationInsightsResourceId: applicationInsightsResourceId
    publicNetworkAccess: 'Enabled'
    managedIdentities: {
      systemAssigned: true
    }
    computes: [
      {
        name: 'cpu-cluster'
        location: location
        sku: 'Standard_DS3_v2'
        scaleSettings: {
          maxNodeCount: 10
          minNodeCount: 0
          nodeIdleTimeBeforeScaleDown: 'PT120S'
        }
        properties: {
          isolatedNetwork: false
          osType: 'Linux'
          remoteLoginPortPublicAccess: 'NotSpecified'
          tier: 'Dedicated'
        }
        computeType: 'AmlCompute'
      }
      {
        name: 'gpu-cluster'
        location: location
        sku: 'Standard_NC6'
        scaleSettings: {
          maxNodeCount: 5
          minNodeCount: 0
          nodeIdleTimeBeforeScaleDown: 'PT120S'
        }
        properties: {
          isolatedNetwork: false
          osType: 'Linux'
          remoteLoginPortPublicAccess: 'NotSpecified'
          tier: 'Dedicated'
        }
        computeType: 'AmlCompute'
      }
    ]
    roleAssignments: [
      {
        principalId: principalId
        roleDefinitionIdOrName: 'Contributor'
        principalType: principalType
      }
    ]
    tags: {
      Environment: environment
      Project: 'Content Recommendation Engine'
      Module: 'AI/ML Services'
    }
  }
}

// Cognitive Search Service using Azure Verified Module
module cognitiveSearchService 'br/public:avm/res/search/search-service:0.6.0' = {
  name: 'cognitiveSearchService'
  params: {
    name: cognitiveSearchName
    location: location
    sku: 'standard'
    replicaCount: 1
    partitionCount: 1
    hostingMode: 'default'
    publicNetworkAccess: 'Enabled'
    authOptions: {
      aadOrApiKey: {
        aadAuthFailureMode: 'http401WithBearerChallenge'
      }
    }
    semanticSearch: 'standard'
    networkRuleSet: {
      ipRules: []
    }
    roleAssignments: [
      {
        principalId: principalId
        roleDefinitionIdOrName: 'Search Service Contributor'
        principalType: principalType
      }
      {
        principalId: principalId
        roleDefinitionIdOrName: 'Search Index Data Contributor'
        principalType: principalType
      }
    ]
    tags: {
      Environment: environment
      Project: 'Content Recommendation Engine'
      Module: 'AI/ML Services'
    }
  }
}

// Cognitive Services Multi-Service Account for additional AI capabilities
module cognitiveServices 'br/public:avm/res/cognitive-services/account:0.7.0' = {
  name: 'cognitiveServices'
  params: {
    name: '${resourcePrefix}-${environment}-cog-${uniqueSuffix}'
    location: location
    kind: 'CognitiveServices'
    sku: 'S0'
    publicNetworkAccess: 'Enabled'
    networkAcls: {
      defaultAction: 'Allow'
    }
    roleAssignments: [
      {
        principalId: principalId
        roleDefinitionIdOrName: 'Cognitive Services Contributor'
        principalType: principalType
      }
    ]
    tags: {
      Environment: environment
      Project: 'Content Recommendation Engine'
      Module: 'AI/ML Services'
    }
  }
}

// Outputs
output machineLearningWorkspaceResourceId string = machineLearningWorkspace.outputs.resourceId
output machineLearningWorkspaceName string = machineLearningWorkspace.outputs.name
output cognitiveSearchServiceResourceId string = cognitiveSearchService.outputs.resourceId
output cognitiveSearchServiceName string = cognitiveSearchService.outputs.name
output cognitiveServicesResourceId string = cognitiveServices.outputs.resourceId
output cognitiveServicesName string = cognitiveServices.outputs.name
