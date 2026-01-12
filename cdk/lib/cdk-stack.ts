import * as cdk from 'aws-cdk-lib/core';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as apigw from 'aws-cdk-lib/aws-apigatewayv2';
import * as integrations from 'aws-cdk-lib/aws-apigatewayv2-integrations';
import * as acm from 'aws-cdk-lib/aws-certificatemanager';
import { Construct } from 'constructs';
import * as path from 'path';

export class CdkStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    const dependenciesLayer = new lambda.LayerVersion(this, 'DependenciesLayer', {
        code: lambda.Code.fromAsset(path.join(__dirname, '../..'), {
            bundling: {
                image: lambda.Runtime.PYTHON_3_12.bundlingImage,
                command: ['bash', '-c', 'bash /asset-input/cdk/scripts/build-layer.sh'],
            },
        }),
        compatibleRuntimes: [lambda.Runtime.PYTHON_3_12],
        description: 'Dependencies for AnthonyFinance Backend',
    });

    const backendLambda = new lambda.Function(this, 'BackendHandler', {
        runtime: lambda.Runtime.PYTHON_3_12,
        code: lambda.Code.fromAsset(path.join(__dirname, '../..'), {
            bundling: {
                image: lambda.Runtime.PYTHON_3_12.bundlingImage,
                command: [
                    'bash', '-c',
                    'cp -r /asset-input/src /asset-output/',
                ],
            },
        }),
        handler: 'src.main.handler',
        timeout: cdk.Duration.seconds(30),
        memorySize: 256,
        layers: [dependenciesLayer],
    })

    // Set these via CDK context: npx cdk deploy -c domain={domain} -c certificateArn={arn:aws:acm:...}
    const domainName = this.node.tryGetContext('domain');
    const certificateArn = this.node.tryGetContext('certificateArn');
    
    let customDomain;
    if (domainName && certificateArn) {
      const certificate = acm.Certificate.fromCertificateArn(this, 'Certificate', certificateArn);
      customDomain = new apigw.DomainName(this, 'CustomDomain', {
        domainName: domainName,
        certificate: certificate,
      });
    }

    const httpApi = new apigw.HttpApi(this, 'BackendApi', {
        apiName: 'AnthonyFinance API',
        ...(customDomain && { defaultDomainMapping: { domainName: customDomain } }),
  });

    httpApi.addRoutes({
        path: '/{proxy+}',
        methods: [apigw.HttpMethod.ANY],
        integration: new integrations.HttpLambdaIntegration('LambdaIntegrations', backendLambda)
    });

    new cdk.CfnOutput(this, 'ApiUrl', {
        value: httpApi.apiEndpoint,
    });

    if (customDomain && domainName) {
      new cdk.CfnOutput(this, 'CustomDomainUrl', {
        value: `https://${domainName}`,
      });
      new cdk.CfnOutput(this, 'DomainNameTarget', {
        value: customDomain.regionalDomainName,
      });
    }
  }
}
