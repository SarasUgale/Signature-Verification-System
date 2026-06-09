const zones = document.querySelectorAll('.drop-zone');

zones.forEach(zone => {

    const input = zone.querySelector('input');
    const img = zone.querySelector('img');

    zone.addEventListener('click',()=>input.click());

    input.addEventListener('change',()=>{

        const file=input.files[0];

        if(file){

            img.src=URL.createObjectURL(file);
            img.style.display='block';

        }

    });

});

const circle=document.getElementById("progressCircle");
const score=document.getElementById("scoreValue");

function animateScore(value){

    const circumference=565;

    const offset=
    circumference-(value/100)*circumference;

    circle.style.strokeDashoffset=offset;

    let current=0;

    const interval=setInterval(()=>{

        current++;

        score.innerText=current+"%";

        if(current>=value){
            clearInterval(interval);
        }

    },20);

}

document
.getElementById("verifyBtn")
.addEventListener("click",verify);

async function verify(){

    const sig1=
    document.getElementById("signature1").files[0];

    const sig2=
    document.getElementById("signature2").files[0];

    if(!sig1 || !sig2){

        alert("Upload both signatures");

        return;
    }

    document.getElementById("loader").style.display="flex";

    const formData=new FormData();

    formData.append("signature1",sig1);
    formData.append("signature2",sig2);

    try{

        const response=
        await fetch("/verify",{
            method:"POST",
            body:formData
        });

        const data=
        await response.json();

        document.getElementById("loader").style.display="none";

        const similarity=
        parseFloat(
        data.message.match(
        /Similarity score: (\d+(\.\d+)?)/
        )[1]
        )*100;

        animateScore(
        Math.round(similarity)
        );

        document.getElementById(
        "statusText"
        ).innerHTML=
        similarity>70
        ? "✅ Authentic Signature"
        : "⚠ Possible Forgery";

        document.getElementById(
        "insightsList"
        ).innerHTML=
        `
        <li>✓ Stroke pattern similarity detected</li>
        <li>✓ Signature geometry analyzed</li>
        <li>✓ Feature vectors compared</li>
        <li>✓ Confidence Score: ${similarity.toFixed(2)}%</li>
        <li>✓ AI Verification Complete</li>
        `;

    }

    catch(err){

        document.getElementById("loader").style.display="none";

        alert("Verification failed");

    }

}