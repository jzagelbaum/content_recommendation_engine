// Parameters
@description('Location for all resources')
param location string

@description('Environment name')
param environment string

@description('Resource prefix')
param resourcePrefix string

@description('Unique suffix for resource naming')
param uniqueSuffix string

@description('Principal ID for role assignments')
param principalId string

@description('Principal type (User or ServicePrincipal)')
@allowed(['User', 'ServicePrincipal'])
param principalType string

// Variables
var keyVaultName = '${resourcePrefix}-${environment}-kv-${uniqueSuffix}'
var logAnalyticsWorkspaceName = '${resourcePrefix}-${environment}-law-${uniqueSuffix}'
var applicationInsightsName = '${resourcePrefix}-${environment}-ai-${uniqueSuffix}'

// Log Analytics Workspace using Azure Verified Module
module logAnalyticsWorkspace 'br/public:avm/res/operational-insights/workspace:0.4.0' = {
  name: 'logAnalyticsWorkspace'
  params: {
    name: logAnalyticsWorkspaceName
    location: location
    skuName: 'PerGB2018'
    dataRetention: 30
    publicNetworkAccessForIngestion: 'Enabled'
    publicNetworkAccessForQuery: 'Enabled'
    tags: {
      Environment: environment
      Project: 'Content Recommendation Engine'
      Module: 'Core Infrastructure'
    }
  }
}

// Application Insights using Azure Verified Module
module applicationInsights 'br/public:avm/res/insights/component:0.4.0' = {
  name: 'applicationInsights'
  params: {
    name: applicationInsightsName
    location: location
    kind: 'web'
    applicationType: 'web'
    workspaceResourceId: logAnalyticsWorkspace.outputs.resourceId
    tags: {
      Environment: environment
      Project: 'Content Recommendation Engine'
      Module: 'Core Infrastructure'
    }
  }
}

// Key Vault using Azure Verified Module
module keyVault 'br/public:avm/res/key-vault/vault:0.8.0' = {
  name: 'keyVault'
  params: {
    name: keyVaultName
    location: location
    sku: 'standard'
    enableVaultForDeployment: true
    enableVaultForTemplateDeployment: true
    enableRbacAuthorization: true
    enableSoftDelete: true
    softDeleteRetentionInDays: 7
    enablePurgeProtection: false // Allow purge for dev environments
    publicNetworkAccess: 'Enabled'
    networkAcls: {
      bypass: 'AzureServices'
      defaultAction: 'Allow'
    }
    diagnosticSettings: [
      {
        workspaceResourceId: logAnalyticsWorkspace.outputs.resourceId
        logCategoriesAndGroups: [
          {
            categoryGroup: 'allLogs'
          }
        ]
        metricCategories: [
          {
            category: 'AllMetrics'
          }
        ]
      }
    ]
    roleAssignments: [
      {
        principalId: principalId
        roleDefinitionIdOrName: 'Key Vault Administrator'
        principalType: principalType
      }
    ]
    tags: {
      Environment: environment
      Project: 'Content Recommendation Engine'
      Module: 'Core Infrastructure'
    }
  }
}

// Container Registry for ML models and custom images
module containerRegistry 'br/public:avm/res/container-registry/registry:0.4.0' = {
  name: 'containerRegistry'
  params: {
    name: '${resourcePrefix}${environment}acr${uniqueSuffix}'
    location: location
    acrSku: 'Basic'
    acrAdminUserEnabled: false
    publicNetworkAccess: 'Enabled'
    anonymousPullEnabled: false
    dataEndpointEnabled: false
    exportPolicyStatus: 'enabled'
    networkRuleBypassOptions: 'AzureServices'
    quarantinePolicyStatus: 'disabled'
    trustPolicyStatus: 'disabled'
    retentionPolicyStatus: 'disabled'
    zoneRedundancy: 'Disabled'
    roleAssignments: [
      {
        principalId: principalId
        roleDefinitionIdOrName: 'AcrPush'
        principalType: principalType
      }
    ]
    diagnosticSettings: [
      {
        workspaceResourceId: logAnalyticsWorkspace.outputs.resourceId
        logCategoriesAndGroups: [
          {
            categoryGroup: 'allLogs'
          }
        ]
        metricCategories: [
          {
            category: 'AllMetrics'
          }
        ]
      }
    ]
    tags: {
      Environment: environment
      Project: 'Content Recommendation Engine'
      Module: 'Core Infrastructure'
    }
  }
}

// Outputs
output keyVaultResourceId string = keyVault.outputs.resourceId
output keyVaultName string = keyVault.outputs.name
output logAnalyticsWorkspaceResourceId string = logAnalyticsWorkspace.outputs.resourceId
output applicationInsightsResourceId string = applicationInsights.outputs.resourceId
output applicationInsightsName string = applicationInsights.outputs.name
output containerRegistryResourceId string = containerRegistry.outputs.resourceId
output containerRegistryName string = containerRegistry.outputs.name
output containerRegistryLoginServer string = containerRegistry.outputs.loginServer
