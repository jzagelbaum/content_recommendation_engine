# Azure Deployment Templates

This directory contains the ARM/JSON templates for the "Deploy to Azure" button functionality.

## Files

### azuredeploy.json
**Purpose**: Main ARM template for one-click Azure deployment

**What it does**:
- Wraps the Bicep templates in ARM JSON format
- Enables the "Deploy to Azure" button functionality
- References the main.bicep template from GitHub
- Defines parameters with user-friendly descriptions
- Provides deployment outputs for post-deployment configuration

**Usage**:
- Automatically used when clicking "Deploy to Azure" button
- Can also be deployed directly via Azure Portal or CLI

**Deploy via Azure CLI**:
```bash
az deployment sub create \
  --location eastus2 \
  --template-file azuredeploy.json \
  --parameters location=eastus2 environment=dev resourcePrefix=contentrec principalId=YOUR_OBJECT_ID
```

### createUiDefinition.json
**Purpose**: Custom UI for Azure Portal deployment experience

**What it does**:
- Creates a step-by-step wizard in Azure Portal
- Provides user-friendly input forms with validation
- Shows helpful descriptions and tooltips
- Displays what will be deployed before confirmation
- Validates inputs (GUID format, character limits, etc.)

**Features**:
- **Basics Tab**: Resource prefix, environment selection
- **Identity & Access Tab**: Azure AD Object ID input with instructions
- **Configuration Tab**: Deployment summary and service list
- **Built-in Validation**: Regex validation, required fields, allowed values

**Wizard Steps**:
1. Basics: Subscription, resource group, prefix, environment
2. Identity: Object ID for role assignments
3. Configuration: Review what will be deployed
4. Review + Create: Final confirmation

### How They Work Together

```
User clicks "Deploy to Azure" button
         ↓
Azure Portal loads createUiDefinition.json
         ↓
User fills out wizard (guided by UI definition)
         ↓
Portal validates inputs
         ↓
User clicks "Review + Create"
         ↓
Portal submits to azuredeploy.json
         ↓
ARM template deploys resources
         ↓
Deployment complete with outputs
```

## Deploy to Azure Button

### Button Syntax

The Deploy to Azure button uses this format:

```markdown
[![Deploy to Azure](https://aka.ms/deploytoazurebutton)](https://portal.azure.com/#create/Microsoft.Template/uri/ENCODED_TEMPLATE_URL/createUIDefinitionUri/ENCODED_UI_URL)
```

### URL Encoding

URLs must be URL-encoded:

**Template URL**:
- Raw: `https://raw.githubusercontent.com/jzagelbaum_microsoft/capstone/main/infrastructure/azuredeploy.json`
- Encoded: `https%3A%2F%2Fraw.githubusercontent.com%2Fjzagelbaum_microsoft%2Fcapstone%2Fmain%2Finfrastructure%2Fazuredeploy.json`

**UI Definition URL**:
- Raw: `https://raw.githubusercontent.com/jzagelbaum_microsoft/capstone/main/infrastructure/createUiDefinition.json`
- Encoded: `https%3A%2F%2Fraw.githubusercontent.com%2Fjzagelbaum_microsoft%2Fcapstone%2Fmain%2Finfrastructure%2FcreateUiDefinition.json`

### Current Button

```markdown
[![Deploy to Azure](https://aka.ms/deploytoazurebutton)](https://portal.azure.com/#create/Microsoft.Template/uri/https%3A%2F%2Fraw.githubusercontent.com%2Fjzagelbaum_microsoft%2Fcapstone%2Fmain%2Finfrastructure%2Fazuredeploy.json/createUIDefinitionUri/https%3A%2F%2Fraw.githubusercontent.com%2Fjzagelbaum_microsoft%2Fcapstone%2Fmain%2Finfrastructure%2FcreateUiDefinition.json)
```

## Testing the Templates

### Test UI Definition Locally

Azure provides a sandbox for testing UI definitions:

1. Go to: https://portal.azure.com/?feature.customPortal=false#blade/Microsoft_Azure_CreateUIDef/SandboxBlade
2. Copy contents of `createUiDefinition.json`
3. Paste into the sandbox
4. Click **Preview** to test the wizard

### Test ARM Template

Validate the ARM template syntax:

```bash
az deployment sub validate \
  --location eastus2 \
  --template-file azuredeploy.json \
  --parameters location=eastus2 environment=dev resourcePrefix=test principalId=00000000-0000-0000-0000-000000000000
```

### Test Full Deployment

Deploy to a test environment:

```bash
az deployment sub create \
  --location eastus2 \
  --template-file azuredeploy.json \
  --parameters @parameters-test.json
```

## Customizing the Templates

### Modify Parameters

Edit `azuredeploy.json` to add/change parameters:

```json
"parameters": {
  "newParameter": {
    "type": "string",
    "defaultValue": "default",
    "metadata": {
      "description": "Description shown in portal"
    },
    "allowedValues": ["value1", "value2"]
  }
}
```

### Modify UI

Edit `createUiDefinition.json` to customize the wizard:

```json
{
  "name": "myControl",
  "type": "Microsoft.Common.TextBox",
  "label": "My Label",
  "toolTip": "Helpful tooltip",
  "constraints": {
    "required": true,
    "regex": "^[a-z0-9]{2,10}$",
    "validationMessage": "Must be 2-10 lowercase characters"
  }
}
```

### Add Wizard Steps

Add new steps to the wizard:

```json
"steps": [
  {
    "name": "myStep",
    "label": "My Step",
    "elements": [
      {
        "name": "myElement",
        "type": "Microsoft.Common.TextBox",
        "label": "My Input"
      }
    ]
  }
]
```

## Common UI Controls

### TextBox
```json
{
  "name": "textInput",
  "type": "Microsoft.Common.TextBox",
  "label": "Text Input",
  "defaultValue": "",
  "toolTip": "Enter text here",
  "constraints": {
    "required": true,
    "regex": "^[a-zA-Z0-9]+$",
    "validationMessage": "Alphanumeric only"
  }
}
```

### DropDown
```json
{
  "name": "dropdown",
  "type": "Microsoft.Common.DropDown",
  "label": "Select Option",
  "defaultValue": "option1",
  "toolTip": "Choose an option",
  "constraints": {
    "allowedValues": [
      {"label": "Option 1", "value": "option1"},
      {"label": "Option 2", "value": "option2"}
    ],
    "required": true
  }
}
```

### InfoBox
```json
{
  "name": "infoBox",
  "type": "Microsoft.Common.InfoBox",
  "visible": true,
  "options": {
    "icon": "Info",
    "text": "Informational message here"
  }
}
```

### Section
```json
{
  "name": "section",
  "type": "Microsoft.Common.Section",
  "label": "Section Title",
  "elements": [
    // Elements inside section
  ]
}
```

## Troubleshooting

### Button Doesn't Work

**Check**:
1. URLs are properly encoded
2. Files are accessible on GitHub (public repo or correct branch)
3. JSON files are valid (no syntax errors)

**Test**:
```bash
# Validate JSON syntax
jq . azuredeploy.json
jq . createUiDefinition.json

# Test URL accessibility
curl https://raw.githubusercontent.com/jzagelbaum_microsoft/capstone/main/infrastructure/azuredeploy.json
```

### UI Definition Errors

**Common issues**:
- Invalid JSON syntax
- Missing required fields
- Incorrect control types
- Invalid regex patterns

**Fix**:
1. Use the [UI Definition Sandbox](https://portal.azure.com/?feature.customPortal=false#blade/Microsoft_Azure_CreateUIDef/SandboxBlade)
2. Check browser console for errors
3. Validate JSON syntax

### Deployment Failures

**Common causes**:
- Missing required parameters
- Invalid parameter values
- Insufficient Azure permissions
- Resource quota limits
- Regional availability

**Debug**:
```bash
# View deployment logs
az deployment sub show \
  --name DEPLOYMENT_NAME \
  --query properties.error

# Check operation details
az deployment operation sub list \
  --name DEPLOYMENT_NAME
```

## Best Practices

### Template Design
- ✅ Use descriptive parameter names
- ✅ Provide helpful descriptions and tooltips
- ✅ Set sensible defaults
- ✅ Use validation (regex, allowed values)
- ✅ Group related parameters
- ✅ Include outputs for important values

### UI Design
- ✅ Keep wizard simple (3-5 steps max)
- ✅ Use clear, non-technical language
- ✅ Provide examples in tooltips
- ✅ Show what will be deployed
- ✅ Validate inputs with helpful messages
- ✅ Use InfoBoxes for important notices

### Security
- ❌ Never include secrets in templates
- ❌ Don't expose sensitive information in outputs
- ✅ Use Key Vault for secrets
- ✅ Use managed identities when possible
- ✅ Follow principle of least privilege

## Resources

- **[ARM Template Documentation](https://docs.microsoft.com/azure/azure-resource-manager/templates/)**
- **[UI Definition Reference](https://docs.microsoft.com/azure/azure-resource-manager/managed-applications/create-uidefinition-overview)**
- **[UI Definition Sandbox](https://portal.azure.com/?feature.customPortal=false#blade/Microsoft_Azure_CreateUIDef/SandboxBlade)**
- **[Deploy to Azure Button Docs](https://docs.microsoft.com/azure/azure-resource-manager/templates/deploy-to-azure-button)**
- **[Bicep Documentation](https://docs.microsoft.com/azure/azure-resource-manager/bicep/)**

## Related Documentation

- **[Deploy to Azure Guide](../docs/deploy-to-azure.md)** - User-facing deployment guide
- **[Infrastructure README](README.md)** - Infrastructure overview
- **[Main README](../README.md)** - Project overview
