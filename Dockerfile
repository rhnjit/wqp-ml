FROM openjdk:8-jdk-alpine
ARG JAR_FILE=*.jar
ARG CFG_FILE=*.properties
COPY ${JAR_FILE} app.jar
COPY ${CFG_FILE} application.properties
COPY ValidationDataset.csv ValidationDataset.csv
COPY TrainingDataset.csv TrainingDataset.csv
ENTRYPOINT ["java","-jar","/app.jar"]