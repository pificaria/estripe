use structopt::StructOpt;
use tokio::net::UdpSocket;
use std::sync::Arc;
use std::str::FromStr;
use std::net::SocketAddrV4;
use parking_lot::RwLock;
use packed_simd::f32x4;

#[derive(StructOpt, Debug)]
#[structopt(name = "estripe")]
struct Opt {
    #[structopt(short, long, default_value = "5510")]
    port: u16,
    #[structopt(short, long, default_value = "estripe")]
    name: String,
}

#[derive(Copy, Clone, Debug)]
struct BiquadCoeff {
    b0: f32,
    cc: f32x4,
}

impl Default for BiquadCoeff {
    fn default() -> BiquadCoeff {
        BiquadCoeff {
            b0: 1.,
            cc: f32x4::new(0., 0., 0., 0.),
        }
    }
}

impl BiquadCoeff {
    pub fn from_bilinear(bv: (f32, f32, f32), av: (f32, f32), w1: f32) -> Self {
        let c = (w1 * 0.5).tan().recip();
        let csq = c*c;
        let r_d = av.1.mul_add(c, av.0 + csq).recip();
        Self {
            b0: r_d * (bv.0 + bv.1*c + bv.2*csq),
            cc: f32x4::new(
                r_d * 2. * bv.2.mul_add(-csq, bv.0),
                r_d * (bv.0 - bv.1*c + bv.2*csq),
                -r_d * 2. * (av.0 - csq),
                -r_d * av.1.mul_add(-c, av.0 + csq),
            ),
        }
    }

    /// Analog resonant lowpass filter
    pub fn analog_resonlp(r_q: f32, w1: f32) -> Self {
        Self::from_bilinear((1., 0., 0.), (1., r_q), w1)
    }

    /// Analog resonant highpass filter
    pub fn analog_resonhp(r_q: f32, w1: f32) -> Self {
        Self::from_bilinear((0., 0., 1.), (1., r_q), w1)
    }

    /// Rb-j peak equalizer
    pub fn rbj_peak(r_q: f32, gain: f32, w1: f32) -> Self {
        let alpha = 0.5 * w1.sin() * r_q;
        let a = 10_f32.powf(gain * 0.025);
        let cos = w1.cos();
        let tmp0 = alpha*a;
        let tmp1 = alpha*a.recip();
        let b0 = 1. + tmp0;
        let b1 = -2. * cos;
        let b2 = 1. - tmp0;
        let r_a0 = (1. + tmp1).recip();
        let a1 = b1;
        let a2 = 1. - tmp1;
        Self {
            b0: r_a0 * b0,
            cc: f32x4::new(
                r_a0 * b1, 
                r_a0 * b2,
                -r_a0 * a1, 
                -r_a0 * a2
            ),
        }
    }

    /// Rb-j lowshelf
    pub fn rbj_lowshelf(r_q: f32, l0: f32, w1: f32) -> Self {
        let alpha = 0.5 * w1.sin() * r_q;
        let a = 10_f32.powf(l0 * 0.025);
        let cos = w1.cos();
        let an1 = a - 1.;
        let ap1 = a + 1.;
        let tmp0 = 2. * a.sqrt() * alpha;
        let b0 = ap1 - an1*cos + tmp0;
        let b1 = 2. * (an1 - ap1*cos);
        let b2 = ap1 - an1*cos - tmp0;
        let r_a0 = (ap1 + an1*cos + tmp0).recip();
        let a1 = -2. * (an1 + ap1*cos);
        let a2 = ap1 + an1*cos - tmp0;
        let tmp1 = r_a0 * a;
        Self {
            b0: tmp1 * b0, 
            cc: f32x4::new(
                tmp1 * b1, 
                tmp1 * b2,
                -r_a0 * a1, 
                -r_a0 * a2
            ),
        }
    }

    /// Rb-j highshelf
    pub fn rbj_highshelf(r_q: f32, lpi: f32, w1: f32) -> Self {
        let alpha = 0.5 * w1.sin() * r_q;
        let a = 10_f32.powf(lpi * 0.025);
        let cos = w1.cos();
        let an1 = a - 1.;
        let ap1 = a + 1.;
        let tmp0 = 2. * a.sqrt() * alpha;
        let b0 = ap1 + an1*cos + tmp0;
        let b1 = -2. * (an1 + ap1*cos);
        let b2 = ap1 + an1*cos - tmp0;
        let r_a0 = (ap1 - an1*cos + tmp0).recip();
        let a1 = 2. * (an1 - ap1*cos);
        let a2 = ap1 - an1*cos - tmp0;
        let tmp1 = r_a0 * a;
        Self {
            b0: tmp1 * b0, 
            cc: f32x4::new(
                tmp1 * b1, 
                tmp1 * b2,
                -r_a0 * a1, 
                -r_a0 * a2
            ),
        }
    }
}

use std::num::FpCategory;
#[inline]
pub fn f32_normalize(x: f32) -> f32 {
    match x.classify() {
        FpCategory::Subnormal|FpCategory::Nan => 0.,
        _ => x
    }
}

#[derive(Default, Debug)]
struct Biquad3 {
    mem: [f32; 8]
}

impl Biquad3 {
    #[inline]
    fn dsp(&mut self, params: &JackParameters, x: f32) -> f32 {
        let r0 = f32_normalize(params.f1_coeff.b0 * x + (params.f1_coeff.cc * f32x4::from_slice_unaligned(&self.mem[..4])).sum());
        let r1 = f32_normalize(params.f2_coeff.b0 * r0 + (params.f2_coeff.cc * f32x4::from_slice_unaligned(&self.mem[2..6])).sum());
        let r2 = f32_normalize(params.f3_coeff.b0 * r1 + (params.f3_coeff.cc * f32x4::from_slice_unaligned(&self.mem[4..])).sum());
        self.mem[1] = self.mem[0]; self.mem[0] = x;
        self.mem[3] = self.mem[2]; self.mem[2] = r0;
        self.mem[5] = self.mem[4]; self.mem[4] = r1;
        self.mem[7] = self.mem[6]; self.mem[6] = r2;
        r2
    }
}

#[derive(Debug)]
struct JackParameters {
    fader: f32,
    mono:  bool,
    f1_coeff: BiquadCoeff,
    f2_coeff: BiquadCoeff,
    f3_coeff: BiquadCoeff,
}

impl Default for JackParameters {
    fn default() -> JackParameters {
        JackParameters {
            fader: 1.,
            mono: false,
            f1_coeff: BiquadCoeff::default(),
            f2_coeff: BiquadCoeff::default(),
            f3_coeff: BiquadCoeff::default(),
        }
    }
}

#[derive(Debug)]
enum FilterType {
    Off,
    LowShelf,
    LowPass,
    Peak,
    HiShelf,
    HiPass,
}

impl Default for FilterType {
    fn default() -> FilterType {
        FilterType::Off
    }
}

impl FromStr for FilterType {
    type Err = ();

    fn from_str(s: &str) -> Result<FilterType, Self::Err> {
        match s.to_lowercase().as_ref() {
            "off" => Ok(FilterType::Off),
            "lowshelf" => Ok(FilterType::LowShelf),
            "lowpass" => Ok(FilterType::LowPass),
            "peak" => Ok(FilterType::Peak),
            "hishelf" => Ok(FilterType::HiShelf),
            "hipass" => Ok(FilterType::HiPass),
            _ => Err(()),
        }
    }
}

#[derive(Debug)]
struct FilterParameters {
    freq: f32,
    q: f32,
    gain: f32, 
    ty: FilterType,
}

impl Default for FilterParameters {
    fn default() -> FilterParameters {
        FilterParameters {
            freq: 90.,
            q: 0.707,
            gain: 0.,
            ty: FilterType::Off,
        }
    }
}

impl FilterParameters {
    fn write_coeff(&self, r_srate: f32, coeff: &mut BiquadCoeff) {
        match self.ty {
            FilterType::Off => {
                coeff.b0 = 1.;
                coeff.cc = f32x4::new(0., 0., 0., 0.);
            },
            FilterType::LowShelf => *coeff = BiquadCoeff::rbj_lowshelf(self.q.recip(), self.gain, freq_to_rad(r_srate, self.freq)),
            FilterType::LowPass => *coeff = BiquadCoeff::analog_resonlp(self.q.recip(), freq_to_rad(r_srate, self.freq)),
            FilterType::HiShelf => *coeff = BiquadCoeff::rbj_highshelf(self.q.recip(), self.gain, freq_to_rad(r_srate, self.freq)),
            FilterType::HiPass => *coeff = BiquadCoeff::analog_resonhp(self.q.recip(), freq_to_rad(r_srate, self.freq)),
            FilterType::Peak => *coeff = BiquadCoeff::rbj_peak(self.q.recip(), self.gain, freq_to_rad(r_srate, self.freq)),
        }
    }
}

#[derive(Default, Debug)]
pub struct Notifications;
impl jack::NotificationHandler for Notifications {}

fn jack_bind(param: Arc<RwLock<JackParameters>>, name: &str, srate: &mut usize) -> jack::AsyncClient<Notifications, jack::ClosureProcessHandler<impl Send + FnMut(&jack::Client, &jack::ProcessScope) -> jack::Control>> {
    let (client, _status) = jack::Client::new(name, jack::ClientOptions::NO_START_SERVER).unwrap();
    let in_left = client.register_port("i_left", jack::AudioIn::default()).unwrap();
    let in_right = client.register_port("i_right", jack::AudioIn::default()).unwrap();
    let mut out_left = client.register_port("o_left", jack::AudioOut::default()).unwrap();
    let mut out_right = client.register_port("o_right", jack::AudioOut::default()).unwrap();
    let param = param.clone();
    let mut fil_left = Biquad3::default();
    let mut fil_right = Biquad3::default();
    let callback = move |_: &jack::Client, ps: &jack::ProcessScope| -> jack::Control {
        let in_left = in_left.as_slice(&ps).into_iter();
        let in_right = in_right.as_slice(&ps).into_iter();
        let out_left = out_left.as_mut_slice(&ps).iter_mut();
        let out_right = out_right.as_mut_slice(&ps).iter_mut();
        let param = param.read();
        for (((&left, &right), o_left), o_right) in in_left.zip(in_right).zip(out_left).zip(out_right) {
            *o_left = fil_left.dsp(&param, param.fader*left); 
            *o_right = fil_right.dsp(&param, param.fader*right);
            // TODO usar um único filtro quando estivermos em mono, pq corta os cálculos do inner
            // loop pela metade.
            if param.mono {
                let out = 0.5 * (*o_left + *o_right);
                *o_left = out;
                *o_right = out;
            } 
        }

        jack::Control::Continue
    };

    let process = jack::ClosureProcessHandler::new(callback);
    *srate = client.sample_rate();
    client.activate_async(Notifications, process).unwrap()
}

#[inline]
pub fn freq_to_rad(r_sr: f32, x: f32) -> f32 {
    2. * std::f32::consts::PI * r_sr * x
}

#[inline]
pub fn rad_to_freq(sr: f32, x: f32) -> f32 {
    0.5 * std::f32::consts::FRAC_1_PI * x * sr
}

#[inline]
pub fn db_to_lin(x: f32) -> f32 {
    10_f32.powf(x * 0.05)
}

#[tokio::main]
async fn main() {
    let opt: Opt = Opt::from_args();
    let addr = SocketAddrV4::from_str(format!("127.0.0.1:{}", opt.port).as_str()).unwrap();
    let osc_sock = UdpSocket::bind(addr).await.unwrap();
    let jack_param = Arc::new(RwLock::new(JackParameters::default()));
    let mut srate: usize = 48000;
    let _jack_client = jack_bind(jack_param.clone(), opt.name.as_str(), &mut srate);
    let mut filters = (FilterParameters::default(), FilterParameters::default(), FilterParameters::default());
    let mut buf = [0u8; rosc::decoder::MTU];
    let r_srate: f32;

    if srate == 0 {
        r_srate = 48000_f32.recip();
    } else {
        r_srate = (srate as f32).recip();
    }

    let osc_server = async move {
        loop {
            let (size, _addr) = osc_sock.recv_from(&mut buf).await.unwrap();
            let packet = rosc::decoder::decode(&buf[..size]).unwrap();
            let mut jack_param = jack_param.write();
            if let rosc::OscPacket::Message(packet) = packet {
                match packet.addr.as_ref() {
                    "/mono" => {
                        if let rosc::OscType::Int(x) = packet.args[0] {
                            jack_param.mono = x != 0;
                        }
                    },
                    "/fader" => {
                        if let rosc::OscType::Float(x) = packet.args[0] {
                            jack_param.fader = db_to_lin(x.min(6.).max(-60.));
                        }
                    },
                    "/fil/1/type" => {
                        if let Some(x) = packet.args[0].clone().string() {
                            filters.0.ty = FilterType::from_str(x.as_str()).unwrap_or(FilterType::Off);
                            filters.0.write_coeff(r_srate, &mut jack_param.f1_coeff);
                        }
                    },
                    "/fil/2/type" => {
                        if let Some(x) = packet.args[0].clone().string() {
                            filters.1.ty = FilterType::from_str(x.as_str()).unwrap_or(FilterType::Off);
                            filters.1.write_coeff(r_srate, &mut jack_param.f2_coeff);
                        }
                    },
                    "/fil/3/type" => {
                        if let Some(x) = packet.args[0].clone().string() {
                            filters.2.ty = FilterType::from_str(x.as_str()).unwrap_or(FilterType::Off);
                            filters.2.write_coeff(r_srate, &mut jack_param.f3_coeff);
                        }
                    },
                    "/fil/1/freq" => {
                        if let rosc::OscType::Float(x) = packet.args[0] {
                            filters.0.freq = x.max(60.).min(10000.);
                            filters.0.write_coeff(r_srate, &mut jack_param.f1_coeff);
                        }
                    },
                    "/fil/2/freq" => {
                        if let rosc::OscType::Float(x) = packet.args[0] {
                            filters.1.freq = x.max(60.).min(10000.);
                            filters.1.write_coeff(r_srate, &mut jack_param.f2_coeff);
                        }
                    },
                    "/fil/3/freq" => {
                        if let rosc::OscType::Float(x) = packet.args[0] {
                            filters.2.freq = x.max(60.).min(10000.);
                            filters.2.write_coeff(r_srate, &mut jack_param.f3_coeff);
                        }
                    },
                    "/fil/1/gain" => {
                        if let rosc::OscType::Float(x) = packet.args[0] {
                            filters.0.gain = x.max(-60.).min(6.);
                            filters.0.write_coeff(r_srate, &mut jack_param.f1_coeff);
                        }
                    },
                    "/fil/2/gain" => {
                        if let rosc::OscType::Float(x) = packet.args[0] {
                            filters.1.gain = x.max(-60.).min(6.);
                            filters.1.write_coeff(r_srate, &mut jack_param.f2_coeff);
                        }
                    },
                    "/fil/3/gain" => {
                        if let rosc::OscType::Float(x) = packet.args[0] {
                            filters.2.gain = x.max(-60.).min(6.);
                            filters.2.write_coeff(r_srate, &mut jack_param.f3_coeff);
                        }
                    },
                    "/fil/1/q" => {
                        if let rosc::OscType::Float(x) = packet.args[0] {
                            filters.0.q = x.max(0.2).min(2.);
                            filters.0.write_coeff(r_srate, &mut jack_param.f1_coeff);
                        }
                    },
                    "/fil/2/q" => {
                        if let rosc::OscType::Float(x) = packet.args[0] {
                            filters.1.q = x.max(0.2).min(2.);
                            filters.1.write_coeff(r_srate, &mut jack_param.f2_coeff);
                        }
                    },
                    "/fil/3/q" => {
                        if let rosc::OscType::Float(x) = packet.args[0] {
                            filters.2.q = x.max(0.2).min(2.);
                            filters.2.write_coeff(r_srate, &mut jack_param.f3_coeff);
                        }
                    },
                    "/dump" => {
                        println!("== estripe: dumping state");
                        println!("   /fader         {}", jack_param.fader);
                        println!("   /mono          {}", jack_param.mono);
                        println!("   /fil/1/type    {:?}", filters.0.ty);
                        println!("   /fil/1/freq    {}", filters.0.freq);
                        println!("   /fil/1/gain    {}", filters.0.gain);
                        println!("   /fil/1/q       {}", filters.0.q);
                        println!("   /fil/2/type    {:?}", filters.1.ty);
                        println!("   /fil/2/freq    {}", filters.1.freq);
                        println!("   /fil/2/gain    {}", filters.1.gain);
                        println!("   /fil/2/q       {}", filters.1.q);
                        println!("   /fil/3/type    {:?}", filters.2.ty);
                        println!("   /fil/3/freq    {}", filters.2.freq);
                        println!("   /fil/3/gain    {}", filters.2.gain);
                        println!("   /fil/3/q       {}", filters.2.q);
                    },
                    _ => (),
                }

                #[cfg(debug_assertions)]
                println!("{:?} {:?} {:?}", jack_param.f1_coeff, jack_param.f2_coeff, jack_param.f3_coeff);
            }
        }
    };

    osc_server.await;
}
