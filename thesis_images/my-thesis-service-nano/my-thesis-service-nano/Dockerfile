FROM openfaas/of-watchdog:0.8.0 as watchdog
FROM python:3.8-slim-buster

RUN mkdir -p /home/app

COPY --from=watchdog /fwatchdog /usr/bin/fwatchdog
RUN chmod +x /usr/bin/fwatchdog

# Add non root user
RUN addgroup --system app && adduser app --system --ingroup app
RUN chown app /home/app

WORKDIR /home/app

USER app

COPY . .

#RUN python measure_time_energy.py 'python 15_linear_regression.py' 2

RUN python3 -m pip install -r requirements.txt

#CMD python measure_time_energy.py "python 15_linear_regression.py" 2

#Populate example here -i.e. "cat", "sha512sum" or "node index.js" 
ENV fprocess="python3 measure_time_energy.py python3 18_tree_regression_compare_2.py 2"

#Set to true to see requests in function logs
ENV write_debug="true"


EXPOSE 8080

HEALTHCHECK --interval=3s CMD [ -e /tmp/.lock ] || exit 1

CMD ["fwatchdog"]
