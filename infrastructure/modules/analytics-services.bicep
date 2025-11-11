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

@description('Data Lake Storage Account Resource ID')
param dataLakeStorageAccountResourceId string

@description('Data Lake File System Name')
param dataLakeFileSystemName string

@description('Principal ID for role assignments')
param principalId string

@description('Principal type (User or ServicePrincipal)')
@allowed(['User', 'ServicePrincipal'])
param principalType string

// Variables
var synapseWorkspaceName = '${resourcePrefix}-${environment}-synapse-${uniqueSuffix}'
var sqlPoolName = '${resourcePrefix}${environment}sqlpool${uniqueSuffix}'
var bigDataPoolName = '${resourcePrefix}${environment}spark${uniqueSuffix}'

// Synapse Analytics Workspace using Azure Verified Module
module synapseWorkspace 'br/public:avm/res/synapse/workspace:0.8.0' = {
  name: 'synapseWorkspace'
  params: {
    name: synapseWorkspaceName
    location: location
    defaultDataLakeStorageAccountResourceId: dataLakeStorageAccountResourceId
    defaultDataLakeStorageFilesystem: dataLakeFileSystemName
    managedVirtualNetwork: true
    preventDataExfiltration: false
    publicNetworkAccess: 'Enabled'
    sqlAdministratorLogin: 'sqladmin'
    sqlAdministratorLoginPassword: 'TempPassword123!' // Should be replaced with Key Vault secret
    managedIdentities: {
      userAssignedResourceIds: []
    }
    integrationRuntimes: [
      {
        name: 'AutoResolveIntegrationRuntime'
        type: 'Managed'
        typeProperties: {
          computeProperties: {
            location: 'AutoResolve'
          }
        }
      }
    ]
    // Removed unsupported sqlPools and bigDataPools properties. Deploy dedicated SQL and Spark pools using separate modules if required.
    roleAssignments: [
      {
        principalId: principalId
        roleDefinitionIdOrName: 'Synapse Administrator'
        principalType: principalType
      }
    ]
    tags: {
      Environment: environment
      Project: 'Content Recommendation Engine'
      Module: 'Analytics Services'
    }
  }
}

// Azure Data Factory for orchestration and data movement
module dataFactory 'br/public:avm/res/data-factory/factory:0.4.0' = {
  name: 'dataFactory'
  params: {
    name: '${resourcePrefix}-${environment}-adf-${uniqueSuffix}'
    location: location
    publicNetworkAccess: 'Enabled'
    managedIdentities: {
      systemAssigned: true
    }
    roleAssignments: [
      {
        principalId: principalId
        roleDefinitionIdOrName: 'Data Factory Contributor'
        principalType: principalType
      }
    ]
    tags: {
      Environment: environment
      Project: 'Content Recommendation Engine'
      Module: 'Analytics Services'
    }
  }
}

// Outputs
output synapseWorkspaceResourceId string = synapseWorkspace.outputs.resourceID
output synapseWorkspaceName string = synapseWorkspace.outputs.name
output dataFactoryResourceId string = dataFactory.outputs.resourceId
output dataFactoryName string = dataFactory.outputs.name
