Dev Branch.

```bash
docker build -t mohdtalal3/flask-perceptron-app .
```

```bash
docker run -p 5000:5000 mohdtalal3/flask-perceptron-app
```

- Run the Jenkins Container:

```bash
docker run --user root -p 8080:8080 -p 5000:5000 -v jenkins_home:/var/jenkins_home -v /var/run/docker.sock:/var/run/docker.sock -d jenkins/jenkins:lts
```
