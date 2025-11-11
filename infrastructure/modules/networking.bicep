// Parameters
@description('Location for all resources')
param location string

@description('Environment name')
param environment string

@description('Resource prefix')
param resourcePrefix string

@description('Unique suffix for resource naming')
param uniqueSuffix string

// Variables
var vnetName = '${resourcePrefix}-${environment}-vnet-${uniqueSuffix}'
var defaultSubnetName = '${resourcePrefix}-${environment}-subnet-default'
var privateEndpointSubnetName = '${resourcePrefix}-${environment}-subnet-pe'
var mlSubnetName = '${resourcePrefix}-${environment}-subnet-ml'
var synapseSubnetName = '${resourcePrefix}-${environment}-subnet-synapse'

// Virtual Network using Azure Verified Module
module virtualNetwork 'br/public:avm/res/network/virtual-network:0.4.0' = {
  name: 'virtualNetwork'
  params: {
    name: vnetName
    location: location
    addressPrefixes: ['10.0.0.0/16']
    subnets: [
      {
        name: defaultSubnetName
        addressPrefix: '10.0.1.0/24'
        serviceEndpoints: [
          'Microsoft.Storage'
          'Microsoft.KeyVault'
        ]
      }
      {
        name: privateEndpointSubnetName
        addressPrefix: '10.0.2.0/24'
        privateEndpointNetworkPolicies: 'Disabled'
        privateLinkServiceNetworkPolicies: 'Disabled'
      }
      {
        name: mlSubnetName
        addressPrefix: '10.0.3.0/24'
        serviceEndpoints: [
          'Microsoft.Storage'
          'Microsoft.KeyVault'
          'Microsoft.ContainerRegistry'
        ]
        delegation: 'Microsoft.MachineLearningServices/workspaces'
      }
      {
        name: synapseSubnetName
        addressPrefix: '10.0.4.0/24'
        serviceEndpoints: [
          'Microsoft.Storage'
          'Microsoft.Sql'
        ]
      }
    ]
    tags: {
      Environment: environment
      Project: 'Content Recommendation Engine'
      Module: 'Networking'
    }
  }
}

// Network Security Group for ML subnet
module mlNetworkSecurityGroup 'br/public:avm/res/network/network-security-group:0.5.0' = {
  name: 'mlNetworkSecurityGroup'
  params: {
    name: '${resourcePrefix}-${environment}-nsg-ml-${uniqueSuffix}'
    location: location
    securityRules: [
      {
        name: 'AllowAzureMLInbound'
        properties: {
          priority: 100
          protocol: 'Tcp'
          access: 'Allow'
          direction: 'Inbound'
          sourceAddressPrefix: 'AzureMachineLearning'
          sourcePortRange: '*'
          destinationAddressPrefix: '*'
          destinationPortRanges: ['29876', '29877']
        }
      }
      {
        name: 'AllowBatchNodeManagement'
        properties: {
          priority: 110
          protocol: 'Tcp'
          access: 'Allow'
          direction: 'Inbound'
          sourceAddressPrefix: 'BatchNodeManagement'
          sourcePortRange: '*'
          destinationAddressPrefix: '*'
          destinationPortRange: '29876-29877'
        }
      }
      {
        name: 'AllowAzureLoadBalancer'
        properties: {
          priority: 120
          protocol: '*'
          access: 'Allow'
          direction: 'Inbound'
          sourceAddressPrefix: 'AzureLoadBalancer'
          sourcePortRange: '*'
          destinationAddressPrefix: '*'
          destinationPortRange: '*'
        }
      }
      {
        name: 'DenyAllInbound'
        properties: {
          priority: 4000
          protocol: '*'
          access: 'Deny'
          direction: 'Inbound'
          sourceAddressPrefix: '*'
          sourcePortRange: '*'
          destinationAddressPrefix: '*'
          destinationPortRange: '*'
        }
      }
      {
        name: 'AllowInternetOutbound'
        properties: {
          priority: 100
          protocol: '*'
          access: 'Allow'
          direction: 'Outbound'
          sourceAddressPrefix: '*'
          sourcePortRange: '*'
          destinationAddressPrefix: 'Internet'
          destinationPortRange: '*'
        }
      }
    ]
    tags: {
      Environment: environment
      Project: 'Content Recommendation Engine'
      Module: 'Networking'
    }
  }
}

// Private DNS Zones for Private Endpoints
var privateDnsZones = [
  'privatelink.blob.${az.environment().suffixes.storage}'
  'privatelink.dfs.${az.environment().suffixes.storage}'
  'privatelink${az.environment().suffixes.keyvaultDns}'
  'privatelink.azureml.ms'
  'privatelink.notebooks.azure.net'
  'privatelink.search.windows.net'
  'privatelink.sql.azuresynapse.net'
  'privatelink.dev.azuresynapse.net'
  'privatelink.azuresynapse.net'
]

module privateDnsZone 'br/public:avm/res/network/private-dns-zone:0.4.0' = [for (dnsZone, index) in privateDnsZones: {
  name: 'privateDnsZone-${index}'
  params: {
    name: dnsZone
    virtualNetworkLinks: [
      {
        name: '${vnetName}-link'
        virtualNetworkResourceId: virtualNetwork.outputs.resourceId
        registrationEnabled: false
      }
    ]
    tags: {
      Environment: environment
      Project: 'Content Recommendation Engine'
      Module: 'Networking'
    }
  }
}]

// Outputs
output virtualNetworkResourceId string = virtualNetwork.outputs.resourceId
output virtualNetworkName string = virtualNetwork.outputs.name
output defaultSubnetResourceId string = virtualNetwork.outputs.subnetResourceIds[0]
output privateEndpointSubnetResourceId string = virtualNetwork.outputs.subnetResourceIds[1]
output mlSubnetResourceId string = virtualNetwork.outputs.subnetResourceIds[2]
output synapseSubnetResourceId string = virtualNetwork.outputs.subnetResourceIds[3]
output privateDnsZoneResourceIds array = [for (dnsZone, index) in privateDnsZones: privateDnsZone[index].outputs.resourceId]
