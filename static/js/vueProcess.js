 var app = new Vue({
            el: '#app',
            data: {
                urls: '',
                processedData: null
            },
            methods: {
                processURLs() {
                    this.$http.post('http://localhost:5000/process', { urls: this.urls.split(',') })
                        .then(response => {
                            this.processedData = response.body;
                        }, error => {
                            console.error(error);
                        });
                }
            }
        });