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

@description('Principal ID for role assignments')
param principalId string

@description('Principal type (User or ServicePrincipal)')
@allowed(['User', 'ServicePrincipal'])
param principalType string

// Variables
var dataLakeStorageAccountName = '${resourcePrefix}${environment}adls${uniqueSuffix}'
var dataLakeFileSystemName = 'contentrec'

// Data Lake Storage Gen2 using Azure Verified Module
module dataLakeStorageAccount 'br/public:avm/res/storage/storage-account:0.14.3' = {
  name: 'dataLakeStorageAccount'
  params: {
    name: dataLakeStorageAccountName
    location: location
    kind: 'StorageV2'
    skuName: 'Standard_LRS'
    accessTier: 'Hot'
    allowBlobPublicAccess: false
    allowCrossTenantReplication: false
    allowSharedKeyAccess: false // Force Azure AD authentication
    enableHierarchicalNamespace: true // Enable hierarchical namespace for Data Lake
    enableNfsV3: false
    enableSftp: false
    minimumTlsVersion: 'TLS1_2'
    supportsHttpsTrafficOnly: true
    publicNetworkAccess: 'Enabled'
    networkAcls: {
      bypass: 'AzureServices'
      defaultAction: 'Allow'
    }
    blobServices: {
      containerDeleteRetentionPolicyEnabled: true
      containerDeleteRetentionPolicyDays: 7
      deleteRetentionPolicyEnabled: true
      deleteRetentionPolicyDays: 7
      changeFeedEnabled: true
      changeFeedRetentionInDays: 7
      versioning: {
        enabled: true
      }
      containers: [
        {
          name: dataLakeFileSystemName
          publicAccess: 'None'
          metadata: {
            Environment: environment
            Purpose: 'Content Recommendation Engine Data Lake'
          }
        }
        {
          name: 'raw-data'
          publicAccess: 'None'
          metadata: {
            Environment: environment
            Purpose: 'Raw content and user interaction data'
          }
        }
        {
          name: 'processed-data'
          publicAccess: 'None'
          metadata: {
            Environment: environment
            Purpose: 'Processed and feature-engineered data'
          }
        }
        {
          name: 'models'
          publicAccess: 'None'
          metadata: {
            Environment: environment
            Purpose: 'ML model artifacts and metadata'
          }
        }
      ]
    }
    roleAssignments: [
      {
        principalId: principalId
        roleDefinitionIdOrName: 'Storage Blob Data Owner'
        principalType: principalType
      }
    ]
    tags: {
      Environment: environment
      Project: 'Content Recommendation Engine'
      Module: 'Storage Infrastructure'
    }
  }
}

// Private Endpoints for Storage Account
module storagePrivateEndpoint 'br/public:avm/res/network/private-endpoint:0.7.1' = {
  name: 'storagePrivateEndpoint'
  params: {
    name: '${dataLakeStorageAccountName}-blob-pe'
    location: location
    subnetResourceId: privateEndpointSubnetResourceId
    privateLinkServiceConnections: [
      {
        name: '${dataLakeStorageAccountName}-blob-pls'
        properties: {
          privateLinkServiceId: dataLakeStorageAccount.outputs.resourceId
          groupIds: ['blob']
        }
      }
    ]
    privateDnsZoneGroup: {
      privateDnsZoneGroupConfigs: [
        {
          name: 'blob'
          privateDnsZoneResourceId: '/subscriptions/${subscription().subscriptionId}/resourceGroups/${resourceGroup().name}/providers/Microsoft.Network/privateDnsZones/privatelink.blob.${az.environment().suffixes.storage}'
        }
      ]
    }
    tags: {
      Environment: environment
      Project: 'Content Recommendation Engine'
      Module: 'Storage Infrastructure'
    }
  }
}

module storageDfsPrivateEndpoint 'br/public:avm/res/network/private-endpoint:0.7.1' = {
  name: 'storageDfsPrivateEndpoint'
  params: {
    name: '${dataLakeStorageAccountName}-dfs-pe'
    location: location
    subnetResourceId: privateEndpointSubnetResourceId
    privateLinkServiceConnections: [
      {
        name: '${dataLakeStorageAccountName}-dfs-pls'
        properties: {
          privateLinkServiceId: dataLakeStorageAccount.outputs.resourceId
          groupIds: ['dfs']
        }
      }
    ]
    privateDnsZoneGroup: {
      privateDnsZoneGroupConfigs: [
        {
          name: 'dfs'
          privateDnsZoneResourceId: '/subscriptions/${subscription().subscriptionId}/resourceGroups/${resourceGroup().name}/providers/Microsoft.Network/privateDnsZones/privatelink.dfs.${az.environment().suffixes.storage}'
        }
      ]
    }
    tags: {
      Environment: environment
      Project: 'Content Recommendation Engine'
      Module: 'Storage Infrastructure'
    }
  }
}

// Outputs
output dataLakeStorageAccountResourceId string = dataLakeStorageAccount.outputs.resourceId
output dataLakeStorageAccountName string = dataLakeStorageAccount.outputs.name
output dataLakeFileSystemName string = dataLakeFileSystemName
