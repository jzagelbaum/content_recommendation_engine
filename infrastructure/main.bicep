targetScope = 'subscription'

// Parameters
@description('Location for all resources')
param location string = 'East US'

@description('Environment name (dev, staging, prod)')
@allowed(['dev', 'staging', 'prod'])
param environment string = 'dev'

@description('Resource prefix for naming conventions')
@minLength(2)
@maxLength(10)
param resourcePrefix string = 'contentrec'

@description('Principal ID of the user/service principal for role assignments')
param principalId string

@description('Principal type (User or ServicePrincipal)')
@allowed(['User', 'ServicePrincipal'])
param principalType string = 'User'

// Variables
var resourceGroupName = '${resourcePrefix}-${environment}-rg'
var uniqueSuffix = substring(uniqueString(subscription().subscriptionId, resourceGroupName), 0, 6)

// Resource Group
resource resourceGroup 'Microsoft.Resources/resourceGroups@2021-04-01' = {
  name: resourceGroupName
  location: location
  tags: {
    Environment: environment
    Project: 'Content Recommendation Engine'
    CreatedBy: 'Azure Verified Modules'
  }
}

// Core Infrastructure Module
module coreInfrastructure 'modules/core-infrastructure.bicep' = {
  scope: resourceGroup
  name: 'core-infrastructure'
  params: {
    location: location
    environment: environment
    resourcePrefix: resourcePrefix
    uniqueSuffix: uniqueSuffix
    principalId: principalId
    principalType: principalType
  }
}

// Networking Module
module networking 'modules/networking.bicep' = {
  scope: resourceGroup
  name: 'networking'
  params: {
    location: location
    environment: environment
    resourcePrefix: resourcePrefix
    uniqueSuffix: uniqueSuffix
  }
}

// Storage Infrastructure Module
module storageInfrastructure 'modules/storage-infrastructure.bicep' = {
  scope: resourceGroup
  name: 'storage-infrastructure'
  params: {
    location: location
    environment: environment
    resourcePrefix: resourcePrefix
    uniqueSuffix: uniqueSuffix
    virtualNetworkResourceId: networking.outputs.virtualNetworkResourceId
    privateEndpointSubnetResourceId: networking.outputs.privateEndpointSubnetResourceId
    principalId: principalId
    principalType: principalType
  }
}

// AI/ML Services Module
module aimlServices 'modules/aiml-services.bicep' = {
  scope: resourceGroup
  name: 'aiml-services'
  params: {
    location: location
    environment: environment
    resourcePrefix: resourcePrefix
    uniqueSuffix: uniqueSuffix
    virtualNetworkResourceId: networking.outputs.virtualNetworkResourceId
    privateEndpointSubnetResourceId: networking.outputs.privateEndpointSubnetResourceId
    keyVaultResourceId: coreInfrastructure.outputs.keyVaultResourceId
    applicationInsightsResourceId: coreInfrastructure.outputs.applicationInsightsResourceId
    storageAccountResourceId: storageInfrastructure.outputs.dataLakeStorageAccountResourceId
    principalId: principalId
    principalType: principalType
  }
}

// Analytics Services Module
module analyticsServices 'modules/analytics-services.bicep' = {
  scope: resourceGroup
  name: 'analytics-services'
  params: {
    location: location
    environment: environment
    resourcePrefix: resourcePrefix
    uniqueSuffix: uniqueSuffix
    virtualNetworkResourceId: networking.outputs.virtualNetworkResourceId
    privateEndpointSubnetResourceId: networking.outputs.privateEndpointSubnetResourceId
    dataLakeStorageAccountResourceId: storageInfrastructure.outputs.dataLakeStorageAccountResourceId
    dataLakeFileSystemName: storageInfrastructure.outputs.dataLakeFileSystemName
    principalId: principalId
    principalType: principalType
  }
}

// OpenAI Services Module
module openaiServices 'modules/openai-services.bicep' = {
  scope: resourceGroup
  name: 'openai-services'
  params: {
    openAIName: '${resourcePrefix}-openai-${uniqueSuffix}'
    location: location
    environmentName: environment
    tags: {
      Environment: environment
      Project: 'Content Recommendation Engine'
      Component: 'OpenAI Services'
    }
  }
}

// OpenAI Function App Module
module openAIFunctionApp 'modules/openai-function-app.bicep' = {
  scope: resourceGroup
  name: 'openai-function-app'
  params: {
    functionAppName: '${resourcePrefix}-openai-func-${uniqueSuffix}'
    location: location
    storageAccountName: '${resourcePrefix}oaifunc${uniqueSuffix}'
    hostingPlanName: '${resourcePrefix}-openai-plan-${uniqueSuffix}'
    applicationInsightsName: '${resourcePrefix}-openai-insights-${uniqueSuffix}'
    keyVaultName: coreInfrastructure.outputs.keyVaultName
    openAIServiceName: openaiServices.outputs.openAIName
    aiSearchServiceName: openaiServices.outputs.aiSearchName
    environment: environment
    tags: {
      Environment: environment
      Project: 'Content Recommendation Engine'
      Component: 'OpenAI Function App'
    }
  }
}

// Outputs
output resourceGroupName string = resourceGroup.name
output virtualNetworkResourceId string = networking.outputs.virtualNetworkResourceId
output dataLakeStorageAccountResourceId string = storageInfrastructure.outputs.dataLakeStorageAccountResourceId
output keyVaultResourceId string = coreInfrastructure.outputs.keyVaultResourceId
output applicationInsightsResourceId string = coreInfrastructure.outputs.applicationInsightsResourceId
output machineLearningWorkspaceResourceId string = aimlServices.outputs.machineLearningWorkspaceResourceId
output cognitiveSearchServiceResourceId string = aimlServices.outputs.cognitiveSearchServiceResourceId
output synapseWorkspaceResourceId string = analyticsServices.outputs.synapseWorkspaceResourceId
output openAIEndpoint string = openaiServices.outputs.openAIEndpoint
output openAIName string = openaiServices.outputs.openAIName
output aiSearchEndpoint string = openaiServices.outputs.aiSearchEndpoint
output aiSearchName string = openaiServices.outputs.aiSearchName
output openAIFunctionAppUrl string = openAIFunctionApp.outputs.functionAppUrl
output openAIFunctionAppName string = openAIFunctionApp.outputs.functionAppName
