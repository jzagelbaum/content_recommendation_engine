@description('Name of the Azure OpenAI service')
param openAIName string

@description('Location for the Azure OpenAI service')
param location string = resourceGroup().location

@description('SKU for the Azure OpenAI service')
param openAISku string = 'S0'

@description('Tags for the resources')
param tags object = {}

@description('Deployment name for GPT model')
param gptDeploymentName string = 'gpt-4'

@description('Deployment name for embedding model')
param embeddingDeploymentName string = 'text-embedding-ada-002'

@description('Environment name for resource naming')
param environmentName string

// Azure OpenAI Account
resource openAIAccount 'Microsoft.CognitiveServices/accounts@2023-05-01' = {
  name: openAIName
  location: location
  tags: tags
  kind: 'OpenAI'
  sku: {
    name: openAISku
  }
  properties: {
    customSubDomainName: openAIName
    publicNetworkAccess: 'Enabled'
    networkAcls: {
      defaultAction: 'Allow'
    }
  }
}

// GPT-4 Deployment
resource gptDeployment 'Microsoft.CognitiveServices/accounts/deployments@2023-05-01' = {
  parent: openAIAccount
  name: gptDeploymentName
  properties: {
    model: {
      format: 'OpenAI'
      name: 'gpt-4'
      version: '0613'
    }
    raiPolicyName: 'Microsoft.Default'
  }
  sku: {
    name: 'Standard'
    capacity: 10
  }
}

// Text Embedding Deployment
resource embeddingDeployment 'Microsoft.CognitiveServices/accounts/deployments@2023-05-01' = {
  parent: openAIAccount
  name: embeddingDeploymentName
  dependsOn: [gptDeployment]
  properties: {
    model: {
      format: 'OpenAI'
      name: 'text-embedding-ada-002'
      version: '2'
    }
    raiPolicyName: 'Microsoft.Default'
  }
  sku: {
    name: 'Standard'
    capacity: 10
  }
}

// Enhanced AI Search for vector storage
resource aiSearchService 'Microsoft.Search/searchServices@2023-11-01' = {
  name: '${openAIName}-search'
  location: location
  tags: tags
  sku: {
    name: environmentName == 'prod' ? 'standard' : 'basic'
  }
  properties: {
    replicaCount: environmentName == 'prod' ? 2 : 1
    partitionCount: 1
    hostingMode: 'default'
    publicNetworkAccess: 'enabled'
    networkRuleSet: {
      ipRules: []
    }
    encryptionWithCmk: {
      enforcement: 'Unspecified'
    }
    disableLocalAuth: false
    authOptions: {
      apiKeyOnly: {}
    }
    semanticSearch: 'free'
  }
}

// Outputs
output openAIEndpoint string = openAIAccount.properties.endpoint
output openAIName string = openAIAccount.name
output openAIId string = openAIAccount.id
output gptDeploymentName string = gptDeployment.name
output embeddingDeploymentName string = embeddingDeployment.name
output aiSearchEndpoint string = 'https://${aiSearchService.name}.search.windows.net'
output aiSearchName string = aiSearchService.name
output aiSearchId string = aiSearchService.id
