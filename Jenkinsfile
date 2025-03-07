pipeline {
    agent any

    environment {
        // Load the Docker Hub credentials stored in Jenkins
        DOCKERHUB_CREDENTIALS = credentials('dockerhub_credentials')
    }

    stages {
        stage('Checkout') {
            steps {
                // Check out the code from the repository
                checkout scm
            }
        }
        stage('Verify Docker') {
            steps {
                sh 'docker --version'
            }
        }
        stage('Build Docker Image') {
            steps {
                script {
                    // Build the Docker image
                    sh "docker build -t mohdtalal3/flask-perceptron-app:latest ."
                }
            }
        }
        stage('Push Docker Image') {
            steps {
                script {
                    // Log in to Docker Hub using the stored credentials
                    sh "echo ${DOCKERHUB_CREDENTIALS_PSW} | docker login -u ${DOCKERHUB_CREDENTIALS_USR} --password-stdin"
                    // Push the Docker image to Docker Hub
                    sh "docker push mohdtalal3/flask-perceptron-app:latest"
                }
            }
        }
    }
    post {
        always {
            // Clean up Docker environment after the job
            sh 'docker rmi mohdtalal3/flask-perceptron-app:latest || true'
           }
        success {
            emailext (
                subject: "Deployment Successful: ${env.JOB_NAME}",
                body: "Deployment on branch ${env.BRANCH_NAME} was successful.",
                to: 'fanasfarooq8888@gmail.com'
            )
        }
        failure {
            emailext (
                subject: "Deployment Failed: ${env.JOB_NAME}",
                body: "Deployment on branch ${env.BRANCH_NAME} has failed.",
                to: 'fanasfarooq8888@gmail.com'
            )
        }
    }
}
