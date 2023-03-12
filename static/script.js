var codeEditor = document.getElementById('code');
// var suggestionButtons = document.querySelectorAll('.suggestion-button');

// suggestionButtons.forEach(function(button) {
//     button.addEventListener('click', function(event) {
//         var suggestionText = event.target.parentNode.parentNode.querySelector('.suggestion-text').textContent;
//         codeEditor.value = suggestionText;
//         codeEditor.rows = suggestionText.split(/\r*\n/).length;
//         var suggestionsList = document.getElementById('suggestions');
//         suggestionsList.innerHTML = '';
//     });
// });

function replaceCode(suggestionText) {
  var codeEditor = document.getElementById('code');
  numLines = suggestionText.split('<br>').length;
  if (numLines ==  codeEditor.rows && numLines > 20){
    codeEditor.rows = numLines+1;
    }
    else{
      codeEditor.rows = 20
    }
  codeEditor.value = suggestionText.replace(/<br>/g, '\n');
}


codeEditor.addEventListener('keydown', function(event) {

  var numLines = codeEditor.value.split(/\r*\n/).length;
  if (numLines ==  codeEditor.rows && numLines > 20){
  codeEditor.rows = numLines+1;
  }
  else{
    codeEditor.rows = 20
  }

  
  if (event.code === "Space" || event.code === 'Enter') {
    console.log('Space bar pressed!');
    var code = codeEditor.value;
    fetch('/suggestions?code=' + encodeURIComponent(code))
      .then(response => response.json())
      .then(data => {
        var suggestions = data['suggestions'];
        var suggestionsList = document.getElementById('suggestions');
        suggestionsList.innerHTML = '';
        for (var i = 0; i < suggestions.length; i++) {
          var suggestion = suggestions[i];
          var suggestionContainer = document.createElement('div');
          suggestionContainer.innerHTML = suggestion;
          suggestionContainer.className = 'container suggestion';
          suggestionsList.appendChild(suggestionContainer);
          
        }
        
      });
  }
});