/**
 * Redirects to given URL
 * @param {*} url 
 * @returns 
 */
const redirect = (url)=> window.location.assign(url)



/**
 * Send POST, PUT, DELETE requests.
 * @param {*} url
 * @param {*} method
 * @param {*} data
 * @param {*} csrftoken
 * @returns
 */
async function postData(url = '', method, data = {}, csrftoken) {
    // Opciones por defecto estan marcadas con un *
    const response = await fetch(url, {
        method: method, // *GET, POST, PUT, DELETE, etc.
        mode: 'cors', // no-cors, *cors, same-origin
        cache: 'no-cache', // *default, no-cache, reload, force-cache, only-if-cached
        credentials: 'same-origin', // include, *same-origin, omit
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': csrftoken
            // 'Content-Type': 'application/x-www-form-urlencoded',
        },
        redirect: 'follow', // manual, *follow, error
        referrerPolicy: 'origin', // no-referrer, *no-referrer-when-downgrade, origin, origin-when-cross-origin, same-origin, strict-origin, strict-origin-when-cross-origin, unsafe-url
        body: JSON.stringify(data) // body data type must match "Content-Type" header
    });
    return response.json(); // parses JSON response into native JavaScript objects

}

