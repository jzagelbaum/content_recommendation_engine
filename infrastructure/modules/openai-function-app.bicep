/*
OpenAI Function App Infrastructure Module
========================================

Creates Azure Function App specifically for OpenAI-powered recommendation system:
- Function App with Python runtime
- Application Insights integration
- Key Vault references for OpenAI credentials
- Storage account for function app
- Service Bus integration for event processing

Author: Content Recommendation Engine Team
Date: October 2025
*/

@description('The name of the OpenAI Function App')
param functionAppName string

@description('The location for all resources')
param location string = resourceGroup().location

@description('The name of the storage account for the function app')
param storageAccountName string

@description('The name of the hosting plan')
param hostingPlanName string

@description('The name of the Application Insights component')
param applicationInsightsName string

@description('The name of the Key Vault containing OpenAI credentials')
param keyVaultName string

@description('The name of the OpenAI service')
param openAIServiceName string

@description('The name of the AI Search service')
param aiSearchServiceName string

@description('Tags for all resources')
param tags object = {}

@description('Environment name (dev, staging, prod)')
param environment string = 'dev'

// Storage account for function app
resource storageAccount 'Microsoft.Storage/storageAccounts@2023-01-01' = {
  name: storageAccountName
  location: location
  tags: tags
  sku: {
    name: 'Standard_LRS'
  }
  kind: 'StorageV2'
  properties: {
    accessTier: 'Hot'
    allowBlobPublicAccess: false
    allowSharedKeyAccess: true
    encryption: {
      services: {
        blob: {
          enabled: true
        }
        file: {
          enabled: true
        }
      }
      keySource: 'Microsoft.Storage'
    }
    networkAcls: {
      defaultAction: 'Allow'
    }
    supportsHttpsTrafficOnly: true
  }
}

// App Service Plan for Function App
resource hostingPlan 'Microsoft.Web/serverfarms@2023-01-01' = {
  name: hostingPlanName
  location: location
  tags: tags
  sku: {
    name: environment == 'prod' ? 'P1V3' : 'Y1'
    tier: environment == 'prod' ? 'Premium' : 'Dynamic'
  }
  kind: 'functionapp'
  properties: {
    reserved: true // Linux
  }
}

// Application Insights
resource applicationInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: applicationInsightsName
  location: location
  tags: tags
  kind: 'web'
  properties: {
    Application_Type: 'web'
    Request_Source: 'rest'
    WorkspaceResourceId: null
  }
}

// Reference existing Key Vault
resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' existing = {
  name: keyVaultName
}

// Reference existing OpenAI service
resource openAIService 'Microsoft.CognitiveServices/accounts@2023-05-01' existing = {
  name: openAIServiceName
}

// Reference existing AI Search service
resource aiSearchService 'Microsoft.Search/searchServices@2023-11-01' existing = {
  name: aiSearchServiceName
}

// Function App
resource functionApp 'Microsoft.Web/sites@2023-01-01' = {
  name: functionAppName
  location: location
  tags: tags
  kind: 'functionapp,linux'
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    serverFarmId: hostingPlan.id
    reserved: true
    httpsOnly: true
    clientAffinityEnabled: false
    siteConfig: {
      linuxFxVersion: 'PYTHON|3.11'
      pythonVersion: '3.11'
      appSettings: [
        {
          name: 'AzureWebJobsStorage'
          value: 'DefaultEndpointsProtocol=https;AccountName=${storageAccount.name};EndpointSuffix=${az.environment().suffixes.storage};AccountKey=${storageAccount.listKeys().keys[0].value}'
        }
        {
          name: 'WEBSITE_CONTENTAZUREFILECONNECTIONSTRING'
          value: 'DefaultEndpointsProtocol=https;AccountName=${storageAccount.name};EndpointSuffix=${az.environment().suffixes.storage};AccountKey=${storageAccount.listKeys().keys[0].value}'
        }
        {
          name: 'WEBSITE_CONTENTSHARE'
          value: toLower(functionAppName)
        }
        {
          name: 'FUNCTIONS_EXTENSION_VERSION'
          value: '~4'
        }
        {
          name: 'FUNCTIONS_WORKER_RUNTIME'
          value: 'python'
        }
        {
          name: 'APPINSIGHTS_INSTRUMENTATIONKEY'
          value: applicationInsights.properties.InstrumentationKey
        }
        {
          name: 'APPLICATIONINSIGHTS_CONNECTION_STRING'
          value: applicationInsights.properties.ConnectionString
        }
        // OpenAI Configuration
        {
          name: 'AZURE_OPENAI_ENDPOINT'
          value: openAIService.properties.endpoint
        }
        {
          name: 'AZURE_OPENAI_API_KEY'
          value: '@Microsoft.KeyVault(VaultName=${keyVault.name};SecretName=openai-api-key)'
        }
        {
          name: 'AZURE_OPENAI_API_VERSION'
          value: '2024-02-01'
        }
        {
          name: 'AZURE_OPENAI_GPT_DEPLOYMENT'
          value: 'gpt-4'
        }
        {
          name: 'AZURE_OPENAI_EMBEDDING_DEPLOYMENT'
          value: 'text-embedding-ada-002'
        }
        // AI Search Configuration
        {
          name: 'AZURE_SEARCH_ENDPOINT'
          value: 'https://${aiSearchService.name}.search.windows.net'
        }
        {
          name: 'AZURE_SEARCH_API_KEY'
          value: '@Microsoft.KeyVault(VaultName=${keyVault.name};SecretName=search-api-key)'
        }
        {
          name: 'AZURE_SEARCH_INDEX_NAME'
          value: 'content-recommendations'
        }
        // Application Configuration
        {
          name: 'ENVIRONMENT'
          value: environment
        }
        {
          name: 'LOG_LEVEL'
          value: environment == 'prod' ? 'INFO' : 'DEBUG'
        }
        {
          name: 'ENABLE_AB_TESTING'
          value: 'true'
        }
        {
          name: 'DEFAULT_AB_TEST_TRAFFIC_SPLIT'
          value: '0.3'
        }
        // Performance and Security Settings
        {
          name: 'WEBSITE_ENABLE_SYNC_UPDATE_SITE'
          value: 'true'
        }
        {
          name: 'WEBSITE_RUN_FROM_PACKAGE'
          value: '1'
        }
        {
          name: 'SCM_DO_BUILD_DURING_DEPLOYMENT'
          value: 'true'
        }
      ]
      cors: {
        allowedOrigins: [
          'https://portal.azure.com'
          'https://ms.portal.azure.com'
        ]
        supportCredentials: false
      }
      ftpsState: 'Disabled'
      http20Enabled: true
      minTlsVersion: '1.2'
      scmMinTlsVersion: '1.2'
      use32BitWorkerProcess: false
      webSocketsEnabled: false
      alwaysOn: environment == 'prod'
      functionAppScaleLimit: environment == 'prod' ? 10 : 5
      minimumElasticInstanceCount: environment == 'prod' ? 1 : 0
    }
  }
}

// Grant Function App access to Key Vault
resource keyVaultAccessPolicy 'Microsoft.KeyVault/vaults/accessPolicies@2023-07-01' = {
  name: 'add'
  parent: keyVault
  properties: {
    accessPolicies: [
      {
        tenantId: functionApp.identity.tenantId
        objectId: functionApp.identity.principalId
        permissions: {
          secrets: [
            'get'
            'list'
          ]
        }
      }
    ]
  }
}

// Grant Function App access to OpenAI service
resource openAIRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: openAIService
  name: guid(openAIService.id, functionApp.id, 'Cognitive Services OpenAI User')
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '5e0bd9bd-7b93-4f28-af87-19fc36ad61bd') // Cognitive Services OpenAI User
    principalId: functionApp.identity.principalId
    principalType: 'ServicePrincipal'
  }
}

// Grant Function App access to AI Search service
resource searchRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: aiSearchService
  name: guid(aiSearchService.id, functionApp.id, 'Search Index Data Contributor')
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '8ebe5a00-799e-43f5-93ac-243d3dce84a7') // Search Index Data Contributor
    principalId: functionApp.identity.principalId
    principalType: 'ServicePrincipal'
  }
}

// Grant Function App read access to storage account
resource storageRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: storageAccount
  name: guid(storageAccount.id, functionApp.id, 'Storage Blob Data Reader')
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '2a2b9908-6ea1-4ae2-8e65-a410df84e7d1') // Storage Blob Data Reader
    principalId: functionApp.identity.principalId
    principalType: 'ServicePrincipal'
  }
}

// Function App deployment slot for staging (only in production)
resource functionAppStaging 'Microsoft.Web/sites/slots@2023-01-01' = if (environment == 'prod') {
  name: 'staging'
  parent: functionApp
  location: location
  tags: tags
  kind: 'functionapp,linux'
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    serverFarmId: hostingPlan.id
    reserved: true
    httpsOnly: true
    clientAffinityEnabled: false
    siteConfig: {
      linuxFxVersion: 'PYTHON|3.11'
      pythonVersion: '3.11'
      appSettings: [
        {
          name: 'AzureWebJobsStorage'
          value: 'DefaultEndpointsProtocol=https;AccountName=${storageAccount.name};EndpointSuffix=${az.environment().suffixes.storage};AccountKey=${storageAccount.listKeys().keys[0].value}'
        }
        {
          name: 'WEBSITE_CONTENTAZUREFILECONNECTIONSTRING'
          value: 'DefaultEndpointsProtocol=https;AccountName=${storageAccount.name};EndpointSuffix=${az.environment().suffixes.storage};AccountKey=${storageAccount.listKeys().keys[0].value}'
        }
        {
          name: 'WEBSITE_CONTENTSHARE'
          value: '${toLower(functionAppName)}-staging'
        }
        {
          name: 'FUNCTIONS_EXTENSION_VERSION'
          value: '~4'
        }
        {
          name: 'FUNCTIONS_WORKER_RUNTIME'
          value: 'python'
        }
        {
          name: 'APPINSIGHTS_INSTRUMENTATIONKEY'
          value: applicationInsights.properties.InstrumentationKey
        }
        {
          name: 'APPLICATIONINSIGHTS_CONNECTION_STRING'
          value: applicationInsights.properties.ConnectionString
        }
        // Same OpenAI and Search configuration as production
        {
          name: 'AZURE_OPENAI_ENDPOINT'
          value: openAIService.properties.endpoint
        }
        {
          name: 'AZURE_OPENAI_API_KEY'
          value: '@Microsoft.KeyVault(VaultName=${keyVault.name};SecretName=openai-api-key)'
        }
        {
          name: 'AZURE_OPENAI_API_VERSION'
          value: '2024-02-01'
        }
        {
          name: 'AZURE_OPENAI_GPT_DEPLOYMENT'
          value: 'gpt-4'
        }
        {
          name: 'AZURE_OPENAI_EMBEDDING_DEPLOYMENT'
          value: 'text-embedding-ada-002'
        }
        {
          name: 'AZURE_SEARCH_ENDPOINT'
          value: 'https://${aiSearchService.name}.search.windows.net'
        }
        {
          name: 'AZURE_SEARCH_API_KEY'
          value: '@Microsoft.KeyVault(VaultName=${keyVault.name};SecretName=search-api-key)'
        }
        {
          name: 'AZURE_SEARCH_INDEX_NAME'
          value: 'content-recommendations-staging'
        }
        {
          name: 'ENVIRONMENT'
          value: 'staging'
        }
        {
          name: 'LOG_LEVEL'
          value: 'DEBUG'
        }
        {
          name: 'ENABLE_AB_TESTING'
          value: 'true'
        }
        {
          name: 'DEFAULT_AB_TEST_TRAFFIC_SPLIT'
          value: '0.5'  // Higher traffic split for staging testing
        }
      ]
      alwaysOn: false
    }
  }
}

// Health check for function app
resource functionAppHealthCheck 'Microsoft.Web/sites/config@2023-01-01' = {
  name: 'web'
  parent: functionApp
  properties: {
    healthCheckPath: '/api/health'
    numberOfWorkers: environment == 'prod' ? 2 : 1
  }
}

// Outputs
output functionAppId string = functionApp.id
output functionAppName string = functionApp.name
output functionAppUrl string = 'https://${functionApp.properties.defaultHostName}'
output functionAppPrincipalId string = functionApp.identity.principalId
output storageAccountId string = storageAccount.id
output storageAccountName string = storageAccount.name
output hostingPlanId string = hostingPlan.id
output applicationInsightsId string = applicationInsights.id
output applicationInsightsInstrumentationKey string = applicationInsights.properties.InstrumentationKey
output stagingSlotUrl string = environment == 'prod' ? 'https://${functionApp.name}-staging.azurewebsites.net' : ''
